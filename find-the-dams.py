# coding=UTF-8
"""Module to process remote dam detection imagery."""
import queue
import threading
import time
import itertools
import pickle
import datetime
import json
import ast
import urllib
import sqlite3
import os
import sys
import logging

import requests
import retrying
import taskgraph
from osgeo import gdal
import shapely.prepared
import shapely.ops
import shapely.wkb
import shapely.strtree
from flask import Flask
import flask

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.INFO)

DAM_BOUNDING_BOX_URL = (
    'https://storage.googleapis.com/natcap-natgeo-dam-ecoshards/'
    'dams_database_md5_7acdf64cd03791126a61478e121c4772.db')
WORLD_BORDERS_VECTOR_PATH = 'world_borders.gpkg'
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace'
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'find-the-dams.db')
PLANET_API_KEY_FILE = 'planet_api_key.txt'
ACTIVE_MOSAIC_JSON_PATH = os.path.join(WORKSPACE_DIR, 'active_mosaic.json')
PLANET_QUADS_DIR = None
REQUEST_TIMEOUT = 5
DATABASE_STATUS_STR = None
STATE_TO_COLOR = {
    'unscheduled': '#666666',
    'fetching data': '#3300FF',
    'analyzing': '#3333FF',
    'complete': '#33CC00',
}

NEXT_STATE = {
    'unscheduled': 'fetching data',
    'fetching data': 'analyzing',
    'analyzing': 'complete',
}

APP = Flask(__name__, static_url_path='', static_folder='')
APP.config['SECRET_KEY'] = b'\xe2\xa9\xd2\x82\xd5r\xef\xdb\xffK\x97\xcfM\xa2WH'


@APP.route('/favicon.ico')
def favicon():
    """Return the favicon for webbrowser decoration."""
    return flask.send_from_directory(
        os.path.join(APP.root_path, 'images'), 'favicon.ico',
        mimetype='image/vnd.microsoft.icon')


@APP.route('/')
def index():
    """Flask entry point."""
    return flask.render_template(
        'dashboard.html', **{
            'message': 'Stats will go here.',
        })


@APP.route('/processing_status/', methods=['POST'])
@retrying.retry(
    wait_exponential_multiplier=1000, wait_exponential_max=10000,
    stop_max_attempt_number=5)
def processing_status():
    """Return results about polygons that are processing."""
    try:
        last_update_tick = int(flask.request.form.get('last_update_tick'))
        LOGGER.debug('last update tick %d', last_update_tick)
        payload = None
        polygons_to_update = {}
        LOGGER.debug(flask.request.form)
        database_uri = 'file:%s?mode=ro' % DATABASE_PATH
        connection = sqlite3.connect(database_uri, uri=True)
        cursor = connection.cursor()

        # get the history that hasn't been sent since the last query
        cursor.execute(
            'SELECT polygon_id_list '
            'FROM update_history '
            'WHERE update_history.update_tick > ?', (last_update_tick,))

        # build a unique set of changed polygons
        polygon_update_id_tuple = tuple(set(itertools.chain(
                *[pickle.loads(blob) for (blob,) in cursor.fetchall()])))
        LOGGER.debug(polygon_update_id_tuple)

        # get the current tick in the database
        (current_update_tick,) = cursor.execute(
            'SELECT max(update_tick) FROM update_history;').fetchone()

        # fetch all polygons that have changed
        cursor.execute(
            "SELECT polygon_id, state, lat_min, lng_min, lat_max, lng_max "
            "FROM spatial_analysis_units "
            "WHERE polygon_id in (" +
            ','.join([str(x) for x in polygon_update_id_tuple]) + ")")

        # construct a return object that indicated which polygons should be
        # updated on the client
        polygons_to_update = {
            polygon_id: {
                'bounds': [[lat_min, lng_min], [lat_max, lng_max]],
                'state': state,
                'color': STATE_TO_COLOR[state]
            } for (
                polygon_id, state, lat_min, lng_min, lat_max, lng_max) in
            cursor.fetchall()
        }

        # count how many polygons just for reference
        cursor.execute(
            "SELECT count(polygon_id) from spatial_analysis_units;")
        (n_processing_units,) = cursor.fetchone()
        # construct final payload
        payload = {
            'last_update_tick': int(current_update_tick),
            'query_time': str(datetime.datetime.now()),
            'n_processing_units': n_processing_units,
            'polygons_to_update': polygons_to_update
        }

        # add all the saptal analysis status
        for state in STATE_TO_COLOR:
            cursor.execute(
                "SELECT count(polygon_id) from spatial_analysis_units "
                "WHERE state=?", (state,))
            payload[state] = cursor.fetchone()[0]

        cursor.close()
        connection.commit()
        return json.dumps(payload)
    except Exception as e:
        LOGGER.exception('encountered exception')
        return str(e)


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def get_bounding_box_quads(
        session, mosaic_quad_list_url, min_x, min_y, max_x, max_y):
    """Query for mosaic via bounding box and retry if necessary."""
    try:
        mosaic_quad_response = session.get(
            f'{mosaic_quad_list_url}?bbox={min_x},{min_y},{max_x},{max_y}',
            timeout=REQUEST_TIMEOUT)
        return mosaic_quad_response
    except Exception:
        LOGGER.exception(
            f"get_bounding_box_quads {min_x},{min_y},{max_x},{max_y} failed")
        raise


def initalize_spatial_search_units(database_path, complete_token_path):
    """Set the initial spatial search units in the database.

    Parameters:
        database_path (str): path to SQLite database that's created by this
            call.
        complete_token_path (str): path to a file that's written if the
            entire initialization process has completed successfully.

    Returns:
        None.

    """
    LOGGER.debug('launching initalize_spatial_search_units')
    create_database_sql = (
        """
        CREATE TABLE update_history (
            update_tick INTEGER NOT NULL PRIMARY KEY,
            polygon_id_list BLOB NOT NULL
        );

        CREATE INDEX update_tick_idx
        ON update_history (update_tick);

        CREATE TABLE spatial_analysis_units (
            polygon_id INTEGER NOT NULL PRIMARY KEY,
            state TEXT NOT NULL,
            country_list TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lng_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lng_max REAL NOT NULL);

        CREATE INDEX sa_lat_min_idx
        ON spatial_analysis_units (lat_min);
        CREATE INDEX sa_lng_min_idx
        ON spatial_analysis_units (lng_min);
        CREATE INDEX sa_lat_max_idx
        ON spatial_analysis_units (lat_max);
        CREATE INDEX sa_lng_max_idx
        ON spatial_analysis_units (lng_max);
        CREATE UNIQUE INDEX sa_id_idx
        ON spatial_analysis_units (polygon_id);

        CREATE TABLE identified_dams (
            dam_id INTEGER NOT NULL PRIMARY KEY,
            pre_known INTEGER NOT NULL,
            dam_description TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lng_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lng_max REAL NOT NULL);

        CREATE INDEX id_lat_min_idx
        ON identified_dams (lat_min);
        CREATE INDEX id_lng_min_idx
        ON identified_dams (lng_min);
        CREATE INDEX id_lat_max_idx
        ON identified_dams (lat_max);
        CREATE INDEX id_lng_max_idx
        ON identified_dams (lng_max);
        CREATE UNIQUE INDEX id_dam_id_idx
        ON identified_dams (dam_id);
        """)
    global DATABASE_STATUS_STR
    DATABASE_STATUS_STR = 'create database tables and indexes'
    LOGGER.debug(DATABASE_STATUS_STR)
    if os.path.exists(database_path):
        os.remove(database_path)
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executescript(create_database_sql)
    cursor.close()
    connection.commit()

    dam_bounding_box_bb_path = os.path.join(
        WORKSPACE_DIR, os.path.basename(DAM_BOUNDING_BOX_URL))
    if not os.path.exists(dam_bounding_box_bb_path):
        dam_bounding_box_bb_tmp_path = '%s.tmp' % dam_bounding_box_bb_path
        DATABASE_STATUS_STR = "download validated dam bounding box database"
        LOGGER.debug(DATABASE_STATUS_STR)
        urllib.request.urlretrieve(
            DAM_BOUNDING_BOX_URL, dam_bounding_box_bb_tmp_path)
        os.rename(dam_bounding_box_bb_tmp_path, dam_bounding_box_bb_path)
    DATABASE_STATUS_STR = "parse dam bounding box database for valid dams"
    LOGGER.debug(DATABASE_STATUS_STR)
    try:
        connection = sqlite3.connect(dam_bounding_box_bb_path)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT "
            "bounding_box_bounds, metadata, validation_table.key, "
            "database_id, description "
            "FROM validation_table "
            "INNER JOIN base_table on base_table.key = validation_table.key;")
        spatial_analysis_unit_list = []
        for payload in cursor:
            (bounding_box_bounds_raw, metadata_raw,
             key, database_id, description) = payload
            metadata = json.loads(metadata_raw)
            if 'checkbox_values' in metadata and (
                    any(metadata['checkbox_values'].values())):
                continue
            if 'comments' in metadata and (
                    'dry' in metadata['comments'].lower()):
                continue
            bounding_box_dict = ast.literal_eval(bounding_box_bounds_raw)
            if bounding_box_dict is None:
                continue
            lat_min, lat_max = list(sorted([
                x['lat'] for x in bounding_box_dict]))
            lng_min, lng_max = list(sorted([
                x['lng'] for x in bounding_box_dict]))
            spatial_analysis_unit_list.append(
                (key, 1, '%s:%s' % (database_id, description), lat_min,
                 lng_min, lat_max, lng_max))
    except Exception:
        LOGGER.exception("Exception encountered.")
    finally:
        cursor.close()
        connection.commit()

    DATABASE_STATUS_STR = (
        "insert valid validated dams into `identified_dams` table")
    LOGGER.debug(DATABASE_STATUS_STR)
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executemany(
        'INSERT INTO identified_dams( '
        'dam_id, pre_known, dam_description, lat_min, lng_min, lat_max, '
        'lng_max) VALUES(?, ?, ?, ?, ?, ?, ?)', spatial_analysis_unit_list)
    cursor.close()
    connection.commit()

    # only search regions that could have dams, i.e. countries and bound by
    # latitude
    DATABASE_STATUS_STR = "convert borders to prepared geometry"
    LOGGER.debug(DATABASE_STATUS_STR)

    world_borders_vector = gdal.OpenEx(
        WORLD_BORDERS_VECTOR_PATH, gdal.OF_VECTOR)
    world_borders_layer = world_borders_vector.GetLayer()
    world_border_polygon_list = []
    for feature in world_borders_layer:
        geom = shapely.wkb.loads(
            feature.GetGeometryRef().ExportToWkb())
        geom.country_name = feature.GetField('NAME')
        world_border_polygon_list.append(geom)

    str_tree = shapely.strtree.STRtree(world_border_polygon_list)
    spatial_analysis_unit_list = []
    current_polygon_id = 0
    # 66N to 56S lat because 66N is arctic circle and 56S is about how far
    # the land went down
    for lat in range(-56, 66):
        LOGGER.debug('processing lat %d', lat)
        for lng in range(-180, 180):
            grid_geom = shapely.geometry.box(lng, lat, lng+1, lat+1)
            name_list = []
            for intersect_geom in str_tree.query(grid_geom):
                if intersect_geom.intersects(grid_geom):
                    name_list.append(intersect_geom.country_name)
            if name_list:
                spatial_analysis_unit_list.append(
                    (current_polygon_id, 'unscheduled',
                     ','.join(name_list), lat, lng, lat+1, lng+1))
                current_polygon_id += 1

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # indicate that the first tick is ALL the polygons being added
    cursor.execute(
        'INSERT INTO update_history(update_tick, polygon_id_list) '
        'VALUES(?, ?)', (0, pickle.dumps(
            list(range(current_polygon_id)))))

    cursor.executemany(
        'INSERT INTO spatial_analysis_units( '
        'polygon_id, state, country_list, lat_min, lng_min, lat_max, lng_max) '
        'VALUES(?, ?, ?, ?, ?, ?, ?)', spatial_analysis_unit_list)
    cursor.close()
    connection.commit()

    with open(complete_token_path, 'w') as token_file:
        token_file.write(str(datetime.datetime.now()))
    DATABASE_STATUS_STR = None


@retrying.retry(
    wait_exponential_multiplier=1000, wait_exponential_max=10000,
    stop_max_attempt_number=5)
def schedule_worker(fetch_work_queue, database_path):
    """Thread to schedule areas to prioritize."""
    try:
        LOGGER.debug('starting schedule worker')
        while True:
            connection = sqlite3.connect(database_path)
            cursor = connection.cursor()
            cursor.execute(
                'SELECT polygon_id FROM spatial_analysis_units '
                'WHERE (country_list like "%South Africa%" '
                'AND state="unscheduled") '
                'OR state="unscheduled" '
                'ORDER BY RANDOM() LIMIT 1;')
            (polygon_id,) = cursor.fetchone()
            LOGGER.debug('scheduling polygon %s', polygon_id)
            cursor.execute(
                'UPDATE spatial_analysis_units SET state=? '
                'WHERE polygon_id=?',
                ('fetching data', polygon_id))
            (current_update_tick,) = cursor.execute(
                'SELECT max(update_tick) FROM update_history;').fetchone()
            cursor.execute(
                'INSERT INTO update_history(update_tick, polygon_id_list) '
                'VALUES(?,?)', (
                    current_update_tick+1, pickle.dumps([polygon_id])))
            cursor.close()
            connection.commit()
            fetch_work_queue.put(polygon_id)
            time.sleep(1)
    except Exception:
        LOGGER.exception('exception in schedule worker')


@retrying.retry(
    wait_exponential_multiplier=1000, wait_exponential_max=10000,
    stop_max_attempt_number=5)
def fetch_worker(
        fetch_queue, download_queue, database_uri, planet_api_key,
        mosaic_quad_list_url):
    """Thread to schedule areas to prioritize."""
    global PLANET_QUADS_DIR
    try:
        LOGGER.debug('starting fetch queue worker')
        session = requests.Session()
        session.auth = (planet_api_key, '')

        while True:
            dam_id = fetch_queue.get()
            LOGGER.debug('about to fetch dam_id %s', dam_id)
            connection = sqlite3.connect(database_uri, uri=True)
            cursor = connection.cursor()
            cursor.execute(
                'SELECT state, lat_min, lng_min, lat_max, lng_max '
                'FROM spatial_analysis_units '
                'WHERE polygon_id=?', (dam_id,))
            (state, lat_min, lng_min, lat_max, lng_max) = cursor.fetchone()
            # find planet bounding boxes
            LOGGER.debug('fetching %s', (lat_min, lng_min, lat_max, lng_max))
            mosaic_quad_response = get_bounding_box_quads(
                session, mosaic_quad_list_url,
                lng_min, lat_min, lng_max, lat_max)
            LOGGER.debug(mosaic_quad_response)
            mosaic_quad_response_dict = mosaic_quad_response.json()
            LOGGER.debug(mosaic_quad_response_dict)
            for mosaic_item in mosaic_quad_response_dict['items']:
                download_url = (mosaic_item['_links']['download'])
                suffix_subdir = os.path.join(
                    *reversed(mosaic_item["id"][-4::]))
                download_raster_path = os.path.join(
                    PLANET_QUADS_DIR, suffix_subdir,
                    f'{mosaic_item["id"]}.tif')
                LOGGER.debug(
                    'download %s to %s', download_url, download_raster_path)
                download_queue.put(
                    (download_url, download_raster_path))
            # download all the tiles that match
            # pass each tile to an inference queue
            LOGGER.debug(
                'ready to fetch %s', (
                    dam_id, state, lat_min, lng_min, lat_max, lng_max))
            cursor.close()
            connection.commit()

            time.sleep(1)
    except Exception:
        LOGGER.exception('exception in fetch worker')


def main():
    """Entry point."""
    try:
        os.makedirs(WORKSPACE_DIR)
    except OSError:
        pass
    initalize_token_path = os.path.join(
        WORKSPACE_DIR, 'initalize_spatial_search_units.COMPLETE')
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1, 5.0)
    task_graph.add_task(
        func=initalize_spatial_search_units,
        args=(DATABASE_PATH, initalize_token_path),
        target_path_list=[initalize_token_path],
        ignore_path_list=[DATABASE_PATH],
        task_name='initialize database')
    task_graph.join()
    fetch_work_queue = queue.Queue(2)
    download_queue = queue.Queue(2)
    schedule_worker_thread = threading.Thread(
        target=schedule_worker,
        args=(fetch_work_queue, DATABASE_PATH,))
    schedule_worker_thread.start()
    ro_database_uri = 'file:%s?mode=ro' % DATABASE_PATH

    # find the most recent mosaic we can use
    with open(PLANET_API_KEY_FILE, 'r') as planet_api_key_file:
        planet_api_key = planet_api_key_file.read().rstrip()

    session = requests.Session()
    session.auth = (planet_api_key, '')

    if not os.path.exists(ACTIVE_MOSAIC_JSON_PATH):
        mosaics_json = session.get(
            'https://api.planet.com/basemaps/v1/mosaics',
            timeout=REQUEST_TIMEOUT)
        most_recent_date = ''
        active_mosaic = None
        for mosaic_data in mosaics_json.json()['mosaics']:
            if mosaic_data['interval'] != '3 mons':
                continue
            last_acquired_date = mosaic_data['last_acquired']
            LOGGER.debug(last_acquired_date)
            if last_acquired_date > most_recent_date:
                most_recent_date = last_acquired_date
                active_mosaic = mosaic_data
        with open(ACTIVE_MOSAIC_JSON_PATH, 'w') as active_mosaic_file:
            active_mosaic_file.write(json.dumps(active_mosaic))
    else:
        with open(ACTIVE_MOSAIC_JSON_PATH, 'r') as active_mosaic_file:
            active_mosaic = json.load(active_mosaic_file)

    global PLANET_QUADS_DIR
    PLANET_QUADS_DIR = os.path.join(
        WORKSPACE_DIR, 'planet_quads_dir', active_mosaic['id'])

    LOGGER.debug(
        'using this mosaic: '
        f"""{active_mosaic['last_acquired']} {active_mosaic['interval']} {
            active_mosaic['grid']['resolution']}""")

    mosaic_quad_list_url = (
        f"""https://api.planet.com/basemaps/v1/mosaics/"""
        f"""{active_mosaic['id']}/quads""")

    fetch_worker_thread = threading.Thread(
        target=fetch_worker,
        args=(
            fetch_work_queue, download_queue, ro_database_uri, planet_api_key,
            mosaic_quad_list_url))
    fetch_worker_thread.start()

    APP.run(host='0.0.0.0', port=8080)
    task_graph.close()
    schedule_worker_thread.join()


if __name__ == '__main__':
    main()
