# coding=UTF-8
r"""This server will pull unclassified quads and attempt to find dams in them.

Launch like this:

sudo docker build docker-image/ -t therealspring/dam-detection:0.0.1

hg fetch && sudo docker run -it --rm -p 80:80 -v \
    `pwd`:/workspace therealspring/dam-detection:0.0.1 \
    "python dam_detector_server.py"

There are x steps:
0) check to see if anything was interrupted on the last run
    * fill this in.
1) grid the planet into chucks that can be processed in parallel
    * commit to database so we don't do it twice
2) schedule fetch of Planet imagery for a grid
    * keep in memory in case there's a fault we can try again
    * schedule may be prioritized by regions, Lisa wants me to do ZA first
3) download imagery
    * check that chunk has not been inferred, if it has, skip
    * commit to database that imagery is downloaded so future scheduling can
      note this
    * TODO: change the grid to show the polygon that has been downloaded
4) fragment imagery into chunks that can be processed by the data inference
   pipeline
    * keep in memory in case there's fault and we can try again, we'll know
      approximately the % of processing for a grid to be done for reporting
    * when a chunk is complete, record:
        * the discovered bounding box and % of confidence for each bb
    * record in database that the chunk has been inferred and need not be
      redownloaded/processed.
    * TODO: change the grid to show the polygon that is being processed for
      for inference


5) when an entire grid is complete

5) search imagery for dams w/ NN inference pipeline or Azure ML service
   pipeline?
    * this will involve breaking larger quads into fragments to analyze
    * these fragments will also be searched for *known* dams
These are the states:
    * unscheduled (static state)
    * scheduled (volatile state)
    * downloaded (static state)
    * inference data on quad (volatile state)
    * complete on quad (static state)

These are the questions we can answer during processing:
    * what is being done right now?
    * how many dams have been discovered?
    * what does a particular dam look like?
    * what dams weren't detected? (i.e. a quad was processed with a dam in it
      that wasn't found)
"""
import argparse
import ast
import datetime
import json
import logging
import multiprocessing
import os
import queue
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import urllib
import uuid

from flask import Flask
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import flask
import numpy
import PIL
import PIL.ImageDraw
import pygeoprocessing
import requests
import retrying
import shapely.geometry
import shapely.ops
import shapely.prepared
import shapely.strtree
import shapely.wkb
import taskgraph
import tensorflow as tf


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
DAM_IMAGE_WORKSPACE = os.path.join(WORKSPACE_DIR, 'identified_dam_imagery')
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'find-the-dams.db')
INITALIZE_DATABASE_TOKEN_PATH = os.path.join(
    WORKSPACE_DIR, 'initalize_spatial_search_units.COMPLETE')
PLANET_API_KEY_FILE = 'planet_api_key.txt'
ACTIVE_MOSAIC_JSON_PATH = os.path.join(WORKSPACE_DIR, 'active_mosaic.json')
REQUEST_TIMEOUT = 5
FRAGMENT_SIZE = (419, 419)  # dims to fragment a Planet quad
THRESHOLD_LEVEL = 0.08
QUERY_LIMIT = 20000  # how many dams to fetch at a time on the app
DETECTOR_POLL_TIME = 3  # seconds to wait between detector complete checks
GLOBAL_HOST_SET = None
GLOBAL_HOST_QUEUE = None
DATABASE_STATUS_STR = None
HOST_FILE_PATH = None
GLOBAL_LOCK = None
WORKING_GRID_ID_STATUS_MAP = None
FRAGMENT_ID_STATUS_MAP = None
SESSION_UUID = None
STATE_TO_COLOR = {
    'unscheduled': '#333333',
    'scheduled': '#FF3333',
    'downloaded': '#FF6600',
    'analyzing': '#6666FF',
    'complete': '#00FF33',
}

DAM_STATE_COLOR = {
    'identified': '#33FF00',
    'pre_known': '#0000F0',
}

APP = Flask(__name__, static_url_path='', static_folder='')


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
        last_known_dam_id = int(flask.request.form.get('last_known_dam_id'))
        payload = None
        polygons_to_update = {}
        LOGGER.debug(flask.request.form)
        database_uri = 'file:%s?mode=ro' % DATABASE_PATH
        connection = sqlite3.connect(database_uri, uri=True)
        cursor = connection.cursor()

        # fetch all grids
        cursor.execute(
            "SELECT grid_id, processing_state, lat_min, lng_min, lat_max, "
            "lng_max FROM grid_status")

        # construct a return object that indicated which polygons should be
        # updated on the client

        polygons_to_update = {
            'grid_%s' % grid_id: {
                'bounds': [[lat_min, lng_min], [lat_max, lng_max]],
                'color': STATE_TO_COLOR[state],
                'fill': 'true' if state == 'unscheduled' else 'false',
                'weight': 1,
            } for (
                grid_id, state, lat_min, lng_min, lat_max, lng_max) in
            cursor.fetchall()
        }

        with GLOBAL_LOCK:
            for grid_id, fragment_info in FRAGMENT_ID_STATUS_MAP.items():
                polygons_to_update[grid_id] = fragment_info

        cursor.execute(
            "SELECT CAST(dam_id as INT) AS dam_id_n, pre_known, "
            "   lat_min, lng_min, lat_max, lng_max "
            "FROM identified_dams "
            "WHERE dam_id_n>? "
            "ORDER BY dam_id_n "
            "LIMIT %d" % QUERY_LIMIT, (last_known_dam_id,))
        dam_count = 0
        max_dam_id = -1
        for dam_id, pre_known, lat_min, lng_min, lat_max, lng_max in cursor:
            dam_count += 1
            max_dam_id = max(max_dam_id, int(dam_id))
            polygons_to_update[dam_id] = {
                'color': (
                    DAM_STATE_COLOR['pre_known'] if pre_known == int(1)
                    else DAM_STATE_COLOR['identified']),
                'bounds': [
                    [lat_min, lng_min],
                    [lat_max, lng_max]],
                'fill': 'false',
                'weight': 1,
            }
        payload = {
            'query_time': str(datetime.datetime.now()),
            'max_dam_id': max_dam_id,
            'polygons_to_update': polygons_to_update,
            'session_uuid': SESSION_UUID,
            'all_sent': 'true' if dam_count < QUERY_LIMIT else 'false',
        }
        cursor.close()
        connection.commit()
        return json.dumps(payload)
    except Exception:
        LOGGER.exception('encountered exception')
        return traceback.format_exc()


def get_bounding_box_quads(
        session, mosaic_quad_list_url, min_x, min_y, max_x, max_y):
    """Query for mosaic via bounding box and retry if necessary.

    Parameters:
        session (requests.Session): an object that is pre-authenticated to
            download given url.
        mosaic_quad_list_url (str): base url to fetch Planet bounding box.
        min_x, min_y, max_x, max_y (float): bounding box coordinates in
            lat/lng.

    Returns:
        list of planet items as described in the API
        https://developers.planet.com/docs/api/reference/#operation/getQuadDownloadLinks

    """
    try:
        items_list = []
        mosaic_quad_response = guarded_session_get(
            session, '%s?bbox=%f,%f,%f,%f' % (
                mosaic_quad_list_url, min_x, min_y, max_x, max_y))

        while True:
            mosaic_quad_dict = mosaic_quad_response.json()
            items_list.extend(mosaic_quad_dict['items'])
            if '_next' in mosaic_quad_dict['_links']:
                mosaic_quad_response = guarded_session_get(
                    session, mosaic_quad_dict['_links']['_next'])
            else:
                break
        return items_list
    except Exception:
        LOGGER.exception(
            "get_bounding_box_quads %f, %f, %f, %f, failed" % (
                min_x, min_y, max_x, max_y))
        raise


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def guarded_session_get(session, url):
    """Get url from session and retry if failure."""
    try:
        return session.get(url, timeout=REQUEST_TIMEOUT)
    except Exception:
        LOGGER.exception("exception on session get for %s", url)
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
        CREATE TABLE grid_status (
            grid_id INTEGER NOT NULL PRIMARY KEY,
            processing_state TEXT NOT NULL,
            country_list TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lng_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lng_max REAL NOT NULL);

        CREATE INDEX sa_lat_min_idx
        ON grid_status (lat_min);
        CREATE INDEX sa_lng_min_idx
        ON grid_status (lng_min);
        CREATE INDEX sa_lat_max_idx
        ON grid_status (lat_max);
        CREATE INDEX sa_lng_max_idx
        ON grid_status (lng_max);
        CREATE UNIQUE INDEX sa_id_idx
        ON grid_status (grid_id);

        CREATE TABLE quad_status (
            quad_id TEXT NOT NULL PRIMARY KEY,
            processing_state TEXT NOT NULL);

        CREATE INDEX sa_quad_id
            ON quad_status (quad_id);
        CREATE INDEX sa_processing_state
            ON quad_status (processing_state);

        CREATE TABLE identified_dams (
            dam_id TEXT NOT NULL,
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
        CREATE INDEX id_dam_id_idx
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
    urllib.request.urlretrieve(
        DAM_BOUNDING_BOX_URL, dam_bounding_box_bb_path)
    DATABASE_STATUS_STR = "parse dam bounding box database for valid dams"
    LOGGER.debug(DATABASE_STATUS_STR)
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
    current_grid_id = 0
    # 66N to 56S lat because 66N is arctic circle and 56S is about how far
    # the land went down
    for lat in range(-56, 66):
        LOGGER.debug('processing lat %d', lat)
        for lng in range(-180, 180):
            grid_geom = shapely.geometry.box(lng, lat, lng + 1, lat + 1)
            name_list = []
            for intersect_geom in str_tree.query(grid_geom):
                if intersect_geom.intersects(grid_geom):
                    name_list.append(intersect_geom.country_name)
            if name_list:
                spatial_analysis_unit_list.append(
                    (current_grid_id, 'unscheduled',
                     ','.join(name_list), lat, lng, lat + 1, lng + 1))
                current_grid_id += 1

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.executemany(
        'INSERT INTO grid_status( '
        'grid_id, processing_state, country_list, '
        'lat_min, lng_min, lat_max, lng_max) '
        'VALUES(?, ?, ?, ?, ?, ?, ?)', spatial_analysis_unit_list)
    cursor.close()
    connection.commit()

    with open(complete_token_path, 'w') as token_file:
        token_file.write(str(datetime.datetime.now()))
    DATABASE_STATUS_STR = None


def schedule_worker(download_work_pipe, readonly_database_uri):
    """Thread to schedule areas to prioritize.

    Parameters:
        download_work_pipe (multiprocessing.Connection): this pipe is used to
            communicate which grid IDs should be scheduled for downloading
            next.
        readonly_database_uri (str): uri to readonly view of Sqlite3 database.

    Returns:
        None.

    """
    try:
        LOGGER.debug('starting schedule worker')
        # iterate through database and check the status of grids, anything
        # that was "scheduled" needs to be re-queued.
        connection = sqlite3.connect(readonly_database_uri, uri=True)
        query_string = (
            'SELECT grid_id FROM grid_status '
            'WHERE processing_state="scheduled" '
            'AND grid_id not in (' + ','.join([
                str(x) for x in WORKING_GRID_ID_STATUS_MAP.keys()]) + ');')
        cursor = connection.cursor()
        cursor.execute(query_string)
        pre_scheduled_id_list = [payload[0] for payload in cursor]
        cursor.close()
        connection.commit()
        for grid_id in pre_scheduled_id_list:
            LOGGER.debug('scheduling grid %s', grid_id)
            download_work_pipe.send(grid_id)
            with GLOBAL_LOCK:
                WORKING_GRID_ID_STATUS_MAP[grid_id] = 'scheduled'
            # wait for ack
            _ = download_work_pipe.recv()

        while True:
            connection = sqlite3.connect(readonly_database_uri, uri=True)
            cursor = connection.cursor()
            # get a random grid but try South Africa first
            query_string = (
                'SELECT grid_id FROM grid_status '
                'WHERE country_list like "%South Africa%" '
                'AND processing_state="unscheduled" '
                'AND grid_id not in (' + ','.join([
                    str(x) for x in WORKING_GRID_ID_STATUS_MAP.keys()]) +
                ') ORDER BY RANDOM() LIMIT 1;')
            LOGGER.debug(query_string)
            cursor.execute(query_string)
            payload = cursor.fetchone()
            if payload:
                grid_id = payload[0]
            else:
                # all the south africa dams have been in process, get one
                # that's not
                cursor.execute(
                    'SELECT grid_id FROM grid_status '
                    'WHERE processing_state="unscheduled" '
                    'AND grid_id not in (' + ','.join([
                        str(x) for x in WORKING_GRID_ID_STATUS_MAP.keys()]) +
                    ') ORDER BY RANDOM() LIMIT 1;')
                (grid_id,) = cursor.fetchone()
            cursor.close()
            connection.commit()
            LOGGER.debug('scheduling grid %s', grid_id)
            download_work_pipe.send(grid_id)
            # wait for acknowledgment
            _ = download_work_pipe.recv()  # noqa: F841
            with GLOBAL_LOCK:
                WORKING_GRID_ID_STATUS_MAP[grid_id] = 'analyzing'
    except Exception:
        LOGGER.exception('exception in schedule worker')
    LOGGER.debug("schedule worker is terminating!")


def download_worker(
        download_worker_pipe, inference_worker_work_queue, database_path,
        planet_api_key, mosaic_quad_list_url, planet_quads_dir):
    """Fetch Planet quads as requested.

    Parameters:
        download_worker_pipe (multiprocessing.Connection): this pipe will
            serve the next grid ID to download quads for.
        inference_worker_work_queue (multiprocessing.Connection): this pipe is
            used to send planet quad id/path tuples for the inference worker.
        database_path (str): path to writable sqlite database.
        planet_api_key (str): key to access Planet's RESTful API.
        mosaic_quad_list_url (str): url that has the Planet global mosaic to
            query for individual quads.
        planet_quads_dir (str): directory to save downloaded planet quads in.
            This function will make tree-like subdirectories under the main
            directory based off the last 3 characters of the quad filename.

    Returns:
        None.

    """
    try:
        LOGGER.debug('starting fetch queue worker')
        session = requests.Session()
        session.auth = (planet_api_key, '')
        download_worker_task_graph = taskgraph.TaskGraph(
            os.path.join(WORKSPACE_DIR, 'download_worker_taskgraph'), -1)

        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        wgs84_wkt = wgs84_srs.ExportToWkt()
        wgs84_srs = None

        while True:
            grid_id = download_worker_pipe.recv()
            # send ack so worker can go to next one
            download_worker_pipe.send('ack')
            LOGGER.debug('about to fetch grid_id %s', grid_id)
            with GLOBAL_LOCK:
                connection = sqlite3.connect(database_path)
                cursor = connection.cursor()
                cursor.execute(
                    'SELECT '
                    'processing_state, lat_min, lng_min, lat_max, lng_max '
                    'FROM grid_status '
                    'WHERE grid_id=?', (grid_id,))
                (processing_state, lat_min, lng_min, lat_max, lng_max) = (
                    cursor.fetchone())
                cursor.execute(
                    'UPDATE grid_status '
                    'SET processing_state="scheduled" '
                    'WHERE grid_id=?',
                    (grid_id,))
                connection.commit()
                cursor.close()
            # find planet bounding boxes
            LOGGER.debug('fetching %s', (lat_min, lng_min, lat_max, lng_max))
            mosaic_item_list = get_bounding_box_quads(
                session, mosaic_quad_list_url,
                lng_min, lat_min, lng_max, lat_max)
            # download all the quads that match
            for mosaic_index, mosaic_item in enumerate(mosaic_item_list):
                download_url = (mosaic_item['_links']['download'])
                suffix_subdir = os.path.join(
                    *reversed(mosaic_item["id"][-4::]))
                download_raster_path = os.path.join(
                    planet_quads_dir, suffix_subdir,
                    '%s.tif' % mosaic_item["id"])

                download_worker_task_graph.add_task(
                    func=download_url_to_file,
                    args=(download_url, download_raster_path),
                    target_path_list=[download_raster_path],
                    task_name='download %s' % os.path.basename(
                        download_raster_path))
                download_worker_task_graph.join()

                # get bounding box for mosaic
                # make a new entry in the FRAGMENT_ID_STATUS_MAP
                raster_info = pygeoprocessing.get_raster_info(
                    download_raster_path)
                raster_wgs84_bb = pygeoprocessing.transform_bounding_box(
                    raster_info['bounding_box'], raster_info['projection'],
                    wgs84_wkt)

                quad_id = '%s_%s' % (grid_id, mosaic_item['id'])
                LOGGER.debug(raster_info)
                with GLOBAL_LOCK:
                    FRAGMENT_ID_STATUS_MAP[quad_id] = {
                        'bounds':
                            [[raster_wgs84_bb[1], raster_wgs84_bb[0]],
                             [raster_wgs84_bb[3], raster_wgs84_bb[2]]],
                        'color': STATE_TO_COLOR['downloaded'],
                        'fill': 'false',
                        'weight': 3,
                    }

                LOGGER.debug('downloaded %s', download_url)
                inference_worker_work_queue.put(
                    (quad_id, download_raster_path))

            with GLOBAL_LOCK:
                connection = sqlite3.connect(database_path)
                cursor = connection.cursor()
                cursor.execute(
                    'UPDATE grid_status '
                    'SET processing_state="scheduled" '
                    'WHERE grid_id=?',
                    (grid_id,))
                cursor.close()
                connection.commit()
    except Exception:
        LOGGER.exception('exception in fetch worker')


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def download_url_to_file(url, target_file_path):
    """Use requests to download a file.

    Parameters:
        url (string): url to file.
        target_file_path (string): local path to download the file.

    Returns:
        None.

    """
    try:
        response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        try:
            os.makedirs(os.path.dirname(target_file_path))
        except OSError:
            pass
        with open(target_file_path, 'wb') as target_file:
            shutil.copyfileobj(response.raw, target_file)
        del response
    except Exception:
        LOGGER.exception(
            "download of %s to %s failed" % (url, target_file_path))
        raise


def inference_worker(
        inference_worker_work_queue, inference_worker_host_queue,
        database_path, worker_id):
    """Take large quads and search for dams.

    Parameters:
        inference_worker_work_queue (multiprocessing.Connection): will get
            ('fragment_id', 'quad_raster_path') tuples where
            'quad_raster_path' is a path to a geotiff that can be searched
            for dam bounding boxes.
        inference_worker_host_queue (queue.Queue): contains a queue of
            host/port strings that can be used as the base to connect to
            the API server. This queue is used to manage multiple servers
            and if they are currently being used/removed/added.
        database_path (str): URI to writable version of database to store
            found dams.
        worker_id (int): a unique ID to identify which worker so we can
            uniquely identify each dam.

    Returns:
        None.

    """
    try:
        with GLOBAL_LOCK:
            connection = sqlite3.connect(database_path)
            cursor = connection.cursor()
            cursor.execute(
                'SELECT max(cast(dam_id as integer)) from identified_dams')
            current_dam_id = int(cursor.fetchone()[0])
            cursor.close()
            connection.commit()

        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        while True:
            quad_id, quad_raster_path = inference_worker_work_queue.get()
            quad_workspace = os.path.join(
                WORKSPACE_DIR, '%s_%s' % (worker_id, quad_id))

            # skip if already calculated
            connection = sqlite3.connect(database_path)
            cursor = connection.cursor()
            cursor.execute(
                'SELECT processing_state FROM quad_status '
                'WHERE quad_id=?', (quad_id,))
            payload = cursor.fetchone()
            if payload is not None:
                processing_state = str(payload[0])
            else:
                processing_state = None
            cursor.close()
            connection.commit()
            if processing_state == 'complete':
                with GLOBAL_LOCK:
                    del FRAGMENT_ID_STATUS_MAP[quad_id]
                LOGGER.info('already completed %s', quad_id)
                continue

            try:
                os.makedirs(quad_workspace)
            except OSError:
                pass

            LOGGER.debug('doing inference on %s', quad_raster_path)
            LOGGER.debug('search for dam bounding boxes in this region')
            with GLOBAL_LOCK:
                FRAGMENT_ID_STATUS_MAP[quad_id]['color'] = (
                    STATE_TO_COLOR['analyzing'])

            raster_info = pygeoprocessing.get_raster_info(quad_raster_path)
            x_size, y_size = raster_info['raster_size']
            dam_list = []
            for i_off in range(0, x_size, FRAGMENT_SIZE[0]):
                for j_off in range(0, y_size, FRAGMENT_SIZE[1]):
                    ul_corner = gdal.ApplyGeoTransform(
                        raster_info['geotransform'], i_off, j_off)
                    # guard against a clip that's too big
                    lr_corner = gdal.ApplyGeoTransform(
                        raster_info['geotransform'],
                        min(i_off+FRAGMENT_SIZE[0], x_size),
                        min(j_off+FRAGMENT_SIZE[1], y_size))
                    xmin, xmax = sorted([ul_corner[0], lr_corner[0]])
                    ymin, ymax = sorted([ul_corner[1], lr_corner[1]])
                    clipped_raster_path = os.path.join(
                        quad_workspace, '%d_%d.tif' % (i_off, j_off))
                    LOGGER.debug("clip to %s", [xmin, ymin, xmax, ymax])
                    pygeoprocessing.warp_raster(
                        quad_raster_path, raster_info['pixel_size'],
                        clipped_raster_path, 'near',
                        target_bb=[xmin, ymin, xmax, ymax])
                    # TODO: get the list of bb coords here and then make
                    # image file somewhere and also make an entry in the
                    # database for image
                    detection_result = do_detection(
                        inference_worker_host_queue, clipped_raster_path,
                        DAM_IMAGE_WORKSPACE, '%s_%s' % (worker_id, quad_id))
                    if detection_result:
                        image_bb_path, coord_list = detection_result
                        for coord_tuple in coord_list:
                            source_srs = osr.SpatialReference()
                            source_srs.ImportFromWkt(raster_info['projection'])
                            to_wgs84 = osr.CoordinateTransformation(
                                source_srs, wgs84_srs)
                            ul_point = ogr.Geometry(ogr.wkbPoint)
                            ul_point.AddPoint(
                                coord_tuple[0][0], coord_tuple[0][1])
                            ul_point.Transform(to_wgs84)
                            lr_point = ogr.Geometry(ogr.wkbPoint)
                            lr_point.AddPoint(
                                coord_tuple[1][0], coord_tuple[1][1])
                            lr_point.Transform(to_wgs84)
                            LOGGER.debug(
                                'found dams at %s %s image %s',
                                str((ul_point.GetX(), ul_point.GetY())),
                                str((lr_point.GetX(), lr_point.GetY())),
                                image_bb_path)
                            dam_list.append((
                                0,
                                'TensorFlow identified dam %d' % (
                                    current_dam_id),
                                lr_point.GetY(), lr_point.GetX(),
                                ul_point.GetY(), ul_point.GetX()))

            with GLOBAL_LOCK:
                connection = sqlite3.connect(database_path)
                cursor = connection.cursor()
                cursor.execute(
                    'SELECT MAX(CAST(dam_id as integer)) '
                    'FROM identified_dams')
                (max_dam_id,) = cursor.fetchone()
                if dam_list:
                    cursor.executemany(
                        'INSERT INTO identified_dams( '
                        'dam_id, pre_known, dam_description, '
                        'lat_min, lng_min, lat_max, '
                        'lng_max) VALUES(?, ?, ?, ?, ?, ?, ?)',
                        [(max_dam_id+index+1,)+dam_tuple
                         for index, dam_tuple in enumerate(dam_list)])
                cursor.execute(
                    'INSERT INTO quad_status (quad_id, processing_state) '
                    'VALUES(?, ?)', (quad_id, "complete"))
                cursor.close()
                connection.commit()

            LOGGER.debug('removing workspace %s', quad_workspace)
            shutil.rmtree(quad_workspace)
            with GLOBAL_LOCK:
                del FRAGMENT_ID_STATUS_MAP[quad_id]
    except Exception:
        LOGGER.exception("Exception in inference_worker")

@retrying.retry(
    wait_exponential_multiplier=1000, wait_exponential_max=10000)
def do_detection(
        inference_worker_host_queue, rgb_raster_path, dam_image_workspace,
        grid_tag):
    """Detect whatever the graph is supposed to detect on a single image.

    Parameters:
        inference_worker_host_queue (queue.Queue): contains a queue of
            host/port strings that can be used as the base to connect to
            the API server. This queue is used to manage multiple servers
            and if they are currently being used/removed/added.
        tf_graph (tensorflow Graph): a loaded graph that can accept
            images of the size in `rgb_raster_path`.
        threshold_level (float): the confidence threshold level to cut off
            classification
        rgb_raster_path (str): path to an RGB geotiff.
        dam_image_workspace (str): path to a directory that can save images.
        grid_tag (str): tag to attach to image file names

    Returns:
        None.

    """
    png_driver = gdal.GetDriverByName('PNG')
    base_raster = gdal.OpenEx(rgb_raster_path, gdal.OF_RASTER)
    png_path = '%s.png' % os.path.splitext(rgb_raster_path)[0]
    png_driver.CreateCopy(png_path, base_raster)
    base_raster = None
    png_driver = None

    image = PIL.Image.open(png_path).convert("RGB")
    height, width = image.size
    image_array = numpy.array(image.getdata()).reshape((height, width, 3))
    print(image_array.shape)

    # ensure we get a valid host
    while True:
        LOGGER.debug('fetching inference worker host')
        inference_worker_host_raw = inference_worker_host_queue.get()
        LOGGER.debug('inference worker host: %s', inference_worker_host_raw)
        with GLOBAL_LOCK:
            # if not this get will have removed the invalid host
            if inference_worker_host_raw in GLOBAL_HOST_SET:
                break

    LOGGER.debug('connecting to %s', inference_worker_host_raw)
    # split off the '?' label if it's there
    inference_worker_host = inference_worker_host_raw.split('?')[0]
    try:
        detect_dam_url = "%s/api/v1/detect_dam" % inference_worker_host

        print('uploading %s to %s' % (rgb_raster_path, detect_dam_url))

        initial_response = requests.post(
            detect_dam_url, timeout=REQUEST_TIMEOUT)
        LOGGER.debug('initial_response: %s', initial_response)
        upload_url = initial_response.json()['upload_url']
        upload_response = requests.put(
            upload_url, files={'file': open(png_path, 'rb')},
            timeout=REQUEST_TIMEOUT)
        LOGGER.debug('upload_response: %s', upload_response)
        status_url = upload_response.json()['status_url']

        while True:
            # keep polling status until error or complete
            status_response = requests.get(
                status_url, timeout=REQUEST_TIMEOUT)
            status_map = status_response.json()
            if status_response.ok:
                LOGGER.debug('status: %s', status_map)
                if status_map['status'] != 'complete':
                    time.sleep(DETECTOR_POLL_TIME)
                else:
                    break
            else:
                LOGGER.error('error: %s', status_map['status'])
                break

        if status_response.ok:
            bb_box_list = status_map['bounding_box_list']
            if not bb_box_list:
                return None
            LOGGER.debug('bounding boxes: %s', bb_box_list)
            download_url = status_map['annotated_png_url']
            png_image_path = os.path.join(
                dam_image_workspace, '%s_%s.png' % (
                    grid_tag, os.path.basename(
                        os.path.splitext(rgb_raster_path)[0])))
            with requests.get(
                    download_url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                with open(png_image_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

            geotransform = pygeoprocessing.get_raster_info(
                rgb_raster_path)['geotransform']
            lat_lng_list = []
            for box in bb_box_list:
                ul_corner = gdal.ApplyGeoTransform(
                    geotransform, float(box[0][0]), float(box[0][1]))
                lr_corner = gdal.ApplyGeoTransform(
                    geotransform, float(box[1][0]), float(box[1][1]))
                lat_lng_list.append((ul_corner, lr_corner))
            return png_image_path, lat_lng_list

        return None
    except Exception:
        LOGGER.exception('something bad happened on do_detection')
        raise
    finally:
        inference_worker_host_queue.put(inference_worker_host_raw)
        LOGGER.debug('done with %s', inference_worker_host_raw)


def load_model(path_to_model):
    """Load a TensorFlow model.

    Parameters:
        path_to_model (str): Path to a tensorflow frozen model.

    Returns:
        TensorFlow graph.

    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            LOGGER.debug('this is the model: %s', path_to_model)
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def main(clear_database):
    """Entry point.

    Parameters:
        clear_database (bool): If true will trigger a reconstruction of the
            database if it doesn't exist.

    """
    for dirname in [WORKSPACE_DIR, DAM_IMAGE_WORKSPACE]:
        try:
            os.makedirs(dirname)
        except OSError:
            pass
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1, 5.0)
    if clear_database:
        initalize_database_task = task_graph.add_task(
            func=initalize_spatial_search_units,
            args=(DATABASE_PATH, INITALIZE_DATABASE_TOKEN_PATH),
            target_path_list=[INITALIZE_DATABASE_TOKEN_PATH],
            ignore_path_list=[DATABASE_PATH],
            task_name='initialize database')
        # wait until the database is initialized before scheduling work
        initalize_database_task.join()

    # go through dam bounding box to put it in global map
    dam_bounding_box_bb_path = os.path.join(
        WORKSPACE_DIR, os.path.basename(DAM_BOUNDING_BOX_URL))
    connection = sqlite3.connect(dam_bounding_box_bb_path)
    cursor = connection.cursor()
    cursor.execute(
        "SELECT "
        "bounding_box_bounds, metadata, validation_table.key, "
        "database_id, description "
        "FROM validation_table "
        "INNER JOIN base_table on base_table.key = validation_table.key;")
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

    cursor.close()
    connection.commit()

    ro_database_uri = 'file:%s?mode=ro' % DATABASE_PATH

    download_worker_pipe, download_scheduler_pipe = multiprocessing.Pipe()
    inference_worker_work_queue = queue.Queue(1)

    schedule_worker_thread = threading.Thread(
        target=schedule_worker,
        args=(download_scheduler_pipe, ro_database_uri))
    schedule_worker_thread.start()

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

    planet_quads_dir = os.path.join(
        WORKSPACE_DIR, 'planet_quads_dir', active_mosaic['id'])

    LOGGER.debug(
        'using this mosaic: %s %s %s' % (
            active_mosaic['last_acquired'], active_mosaic['interval'],
            active_mosaic['grid']['resolution']))

    mosaic_quad_list_url = (
        "https://api.planet.com/basemaps/v1/mosaics/%s/quads" % (
            active_mosaic['id']))

    download_worker_thread = threading.Thread(
        target=download_worker,
        args=(
            download_worker_pipe, inference_worker_work_queue, DATABASE_PATH,
            planet_api_key, mosaic_quad_list_url, planet_quads_dir))
    download_worker_thread.start()

    inference_worker_host_queue = queue.Queue()
    host_file_monitor_thread = threading.Thread(
        target=host_file_monitor,
        args=(HOST_FILE_PATH, inference_worker_host_queue,
              inference_worker_work_queue))
    host_file_monitor_thread.start()

    APP.run(host='0.0.0.0', port=80)
    task_graph.close()
    schedule_worker_thread.join()


def host_file_monitor(
        inference_host_file_path, inference_worker_host_queue,
        inference_worker_work_queue):
    """Watch inference_host_file_path & update inference_worker_host_queue.

    Parameters:
        inference_host_file_path (str): path to a file that contains lines
            of http://[host]:[port]<?label> that can be used to send inference
            work to. <label> can be used to use the same machine more than
            once.
        inference_worker_host_queue (queue.Queue): new hosts are queued here
            so they can be pulled by other workers later.
        inference_worker_work_queue


    """
    last_modified_time = 0
    worker_id = -1
    while True:
        try:
            current_modified_time = os.path.getmtime(inference_host_file_path)
            if current_modified_time != last_modified_time:
                last_modified_time = current_modified_time
                with open(inference_host_file_path, 'r') as ip_file:
                    ip_file_contents = ip_file.readlines()
                with GLOBAL_LOCK:
                    global GLOBAL_HOST_SET
                    old_host_set = GLOBAL_HOST_SET
                    GLOBAL_HOST_SET = set([
                        line.strip() for line in ip_file_contents
                        if line.startswith('http')])
                    new_hosts = GLOBAL_HOST_SET.difference(old_host_set)
                    for new_host in new_hosts:
                        inference_worker_host_queue.put(new_host)
                    n_hosts = len(new_hosts)
                for _ in range(n_hosts):
                    worker_id += 1
                    LOGGER.info(
                        'starting new inference_worker host %d', worker_id)
                    inference_worker_thread = threading.Thread(
                        target=inference_worker,
                        args=(inference_worker_work_queue,
                              inference_worker_host_queue,
                              DATABASE_PATH, worker_id))
                    inference_worker_thread.start()
            time.sleep(DETECTOR_POLL_TIME)
        except Exception:
            LOGGER.exception('exception in `host_file_monitor`')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Start dam detection server.')
    parser.add_argument(
        'host_file', help=(
            'Path to list of host fqdn/ip/port for inference workers '
            'running the natcap/dam-inference-server docker container'))
    parser.add_argument(
        '--clear_database', action='store_true',
        help='backs up the dam database and starts over')

    args = parser.parse_args()
    HOST_FILE_PATH = args.host_file
    if not os.path.exists(HOST_FILE_PATH):
        raise ValueError('%s does not exist', HOST_FILE_PATH)

    if args.clear_database:
        os.remove(INITALIZE_DATABASE_TOKEN_PATH)
        db_base_path, db_extension = os.path.splitext(DATABASE_PATH)
        timestamp_string = (
            str(datetime.datetime.now()).replace(
                ' ', '_').replace(':', '_').replace('.', '_'))
        db_backup_path = '%s_%s%s' % (
            db_base_path, timestamp_string, db_extension)
        LOGGER.warn('moving old database to %s' % db_backup_path)
        shutil.copyfile(
            DATABASE_PATH, db_backup_path)

    GLOBAL_LOCK = threading.Lock()
    GLOBAL_HOST_SET = set()
    GLOBAL_HOST_QUEUE = set()
    WORKING_GRID_ID_STATUS_MAP = {}
    FRAGMENT_ID_STATUS_MAP = {}
    SESSION_UUID = uuid.uuid4().hex
    main(args.clear_database)
