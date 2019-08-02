# coding=UTF-8
"""Module to process remote dam detection imagery."""
import datetime
import json
import ast
import urllib
import sqlite3
import os
import sys
import logging

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
DATABASE_STATUS_STR = None
STATE_TO_COLOR = {
    'unscheduled': '#666666',
    'scheduled': '#FF9900',
    'fetching data': '#3300FF',
    'analyzing': '#3333FF',
    'complete': '#33CC00',
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
def processing_status():
    """Return results about polygons that are processing."""
    try:
        polygons_to_update = {}
        payload = None
        LOGGER.debug(flask.request.form)
        database_uri = 'file:%s?mode=rw' % DATABASE_PATH
        connection = sqlite3.connect(database_uri, uri=True)
        cursor = connection.cursor()
        if flask.request.form.get('update_only') == 'false':
            cursor.execute(
                "SELECT id, state, lat_min, lng_min, lat_max, lng_max "
                "FROM spatial_analysis_units;")
            polygons_to_update = {
                polygon_id: {
                    'bounds': [[lat_min, lng_min], [lat_max, lng_max]],
                    'state': state,
                    'color': STATE_TO_COLOR[state]
                } for (
                    polygon_id, state, lat_min, lng_min, lat_max, lng_max) in
                cursor.fetchall()
            }
        else:
            cursor.execute(
                'SELECT id FROM spatial_analysis_units '
                'ORDER BY RANDOM() LIMIT 1;')
            (polygon_id,) = cursor.fetchone()
            cursor.execute(
                'UPDATE spatial_analysis_units SET state = ? WHERE id = ?',
                ('scheduled', polygon_id))
            cursor.execute(
                'SELECT id, state, lat_min, lng_min, lat_max, lng_max '
                'FROM spatial_analysis_units WHERE id=?', (polygon_id,))
            (polygon_id, state, lat_min, lng_min, lat_max, lng_max) = (
                cursor.fetchone())
            polygons_to_update = {
                polygon_id: {
                    'bounds': [[lat_min, lng_min], [lat_max, lng_max]],
                    'state': state,
                    'color': STATE_TO_COLOR[state]
                }
            }

        cursor.execute("SELECT count(id) from spatial_analysis_units;")
        (n_processing_units,) = cursor.fetchone()
        payload = json.dumps({
                'query_time': str(datetime.datetime.now()),
                'n_processing_units': n_processing_units,
                'polygons_to_update': polygons_to_update
            })
        cursor.close()
        connection.commit()
    except Exception:
        LOGGER.exception('encountered exception')
    finally:
        return payload


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
        CREATE TABLE spatial_analysis_units (
            id INTEGER NOT NULL PRIMARY KEY,
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
        ON spatial_analysis_units (id);

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

    # define the spatial search units
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
    spatial_analysis_id = 0
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
                    (spatial_analysis_id, 'unscheduled',
                     ','.join(name_list), lat, lng, lat+1, lng+1))
                spatial_analysis_id += 1

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executemany(
        'INSERT INTO spatial_analysis_units( '
        'id, state, country_list, lat_min, lng_min, lat_max, lng_max) '
        'VALUES(?, ?, ?, ?, ?, ?, ?)', spatial_analysis_unit_list)
    cursor.close()
    connection.commit()

    with open(complete_token_path, 'w') as token_file:
        token_file.write(str(datetime.datetime.now()))
    DATABASE_STATUS_STR = None


def schedule_workder():
    """Thread to schedule areas to prioritize."""
    pass


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
    APP.run(host='0.0.0.0', port=8080)
    task_graph.close()


if __name__ == '__main__':
    main()
