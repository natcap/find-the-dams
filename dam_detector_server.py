# coding=UTF-8
"""This server will pull unclassified tiles and attempt to find dams in them.

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
    * this will involve breaking larger tiles into fragments to analyze
    * these fragments will also be searched for *known* dams
These are the states:
    * unscheduled (static state)
    * scheduled (volatile state)
    * downloaded (static state)
    * inference data on tile (volatile state)
    * complete on tile (static state)

These are the questions we can answer during processing:
    * what is being done right now?
    * how many dams have been discovered?
    * what does a particular dam look like?
    * what dams weren't detected? (i.e. a tile was processed with a dam in it
      that wasn't found)
"""
import uuid
import shutil
import queue
import threading
import datetime
import json
import ast
import urllib
import sqlite3
import os
import sys
import logging

import ecoshard
import numpy
import pygeoprocessing
import requests
import retrying
import taskgraph
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import shapely.geometry
import shapely.ops
import shapely.prepared
import shapely.strtree
import shapely.wkb
from flask import Flask
import flask
import tensorflow as tf
import PIL
import PIL.ImageDraw


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
INFERENCE_MODEL_URL = (
    'https://storage.googleapis.com/natcap-natgeo-dam-ecoshards/'
    'fasterRCNN_08-26-withnotadams_md5_83f58894e34e1e785fcaa2dbc1d3ec7a.pb')
WORLD_BORDERS_VECTOR_PATH = 'world_borders.gpkg'
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace'
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'find-the-dams.db')
PLANET_API_KEY_FILE = 'planet_api_key.txt'
ACTIVE_MOSAIC_JSON_PATH = os.path.join(WORKSPACE_DIR, 'active_mosaic.json')
REQUEST_TIMEOUT = 5
DICE_SIZE = (419, 419)  # how to dice global quads.
THRESHOLD_LEVEL = 0.08
DATABASE_STATUS_STR = None
GLOBAL_LOCK = None
WORKING_GRID_ID_STATUS_MAP = None
FRAGMENT_ID_STATUS_MAP = None
IDENTIFIED_DAMS = None
SESSION_UUID = None
STATE_TO_COLOR = {
    'unscheduled': '#333333',
    'scheduled': '#FF3333',
    'downloaded': '#FF6600',
    'analyzing': '#6666FF',
    'complete': '#00FF33',
}

NEXT_STATE = {
    'unscheduled': 'scheduled',
    'scheduled': 'downloaded',
    'downloaded': 'analyzing',
    'analyzing': 'complete',
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
        client_uuid = str(flask.request.form.get('session_uuid'))
        new_session = False
        if client_uuid != SESSION_UUID:
            # we have a new session, reset the remote
            LOGGER.debug(
                '%s is not equal to %s uuid so resetting remote',
                client_uuid, SESSION_UUID)
            new_session = True
        payload = None
        polygons_to_update = {}
        LOGGER.debug(flask.request.form)
        database_uri = 'file:%s?mode=ro' % DATABASE_PATH
        connection = sqlite3.connect(database_uri, uri=True)
        cursor = connection.cursor()

        if new_session:
            # fetch all grids
            cursor.execute(
                "SELECT grid_id, processing_state, lat_min, lng_min, lat_max, "
                "lng_max FROM grid_status")

            # construct a return object that indicated which polygons should be
            # updated on the client
            polygons_to_update = {
                grid_id: {
                    'bounds': [[lat_min, lng_min], [lat_max, lng_max]],
                    'color': STATE_TO_COLOR[state]
                } for (
                    grid_id, state, lat_min, lng_min, lat_max, lng_max) in
                cursor.fetchall()
            }
        else:
            polygons_to_update = {}

        with GLOBAL_LOCK:
            for grid_id, fragment_info in FRAGMENT_ID_STATUS_MAP.items():
                polygons_to_update[grid_id] = fragment_info

            for dam_id, dam_info in IDENTIFIED_DAMS.items():
                LOGGER.debug(dam_info)
                polygons_to_update[dam_id] = dam_info
            for grid_id, status in WORKING_GRID_ID_STATUS_MAP.items():
                if grid_id in polygons_to_update:
                    polygons_to_update[grid_id]['color'] = (
                        STATE_TO_COLOR[status])
                else:
                    polygons_to_update[grid_id] = {
                        'color': STATE_TO_COLOR[status]
                    }

        # count how many polygons just for reference
        cursor.execute(
            "SELECT count(grid_id) from grid_status;")
        (n_processing_units,) = cursor.fetchone()
        # construct final payload
        payload = {
            'query_time': str(datetime.datetime.now()),
            'n_processing_units': n_processing_units,
            'polygons_to_update': polygons_to_update,
            'session_uuid': SESSION_UUID,
        }

        # add all the spatial analysis status
        for processing_state in STATE_TO_COLOR:
            cursor.execute(
                "SELECT count(grid_id) from grid_status "
                "WHERE processing_state=?", (processing_state,))
            payload[processing_state] = cursor.fetchone()[0]

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
            '%s?bbox=%f,%f,%f,%f' % (
                mosaic_quad_list_url, min_x, min_y, max_x, max_y),
            timeout=REQUEST_TIMEOUT)
        return mosaic_quad_response
    except Exception:
        LOGGER.exception(
            "get_bounding_box_quads %f, %f, %f, %f, failed" % (
                min_x, min_y, max_x, max_y))
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
    current_grid_id = 0
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
                    (current_grid_id, 'unscheduled',
                     ','.join(name_list), lat, lng, lat+1, lng+1))
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


def schedule_worker(download_work_queue, readonly_database_uri):
    """Thread to schedule areas to prioritize.

    Parameters:
        download_work_queue (queue): this queue is used to communicate which
            grid IDs should be scheduled for downloading next
        readonly_database_uri (str): uri to readonly view of Sqlite3 database.

    Returns:
        None.

    """
    try:
        LOGGER.debug('starting schedule worker')
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
            LOGGER.debug('scheduling grid %s', grid_id)
            download_work_queue.put(grid_id)
            with GLOBAL_LOCK:
                WORKING_GRID_ID_STATUS_MAP[grid_id] = 'scheduled'
            cursor.close()
            connection.commit()
            # this will only do the first grid
            break
    except Exception:
        LOGGER.exception('exception in schedule worker')
    LOGGER.debug("schedule worker is terminating!")


def download_worker(
        download_queue, inference_queue, database_uri, planet_api_key,
        mosaic_quad_list_url, planet_quads_dir):
    """Fetch Planet tiles as requested.

    Parameters:
        download_queue (queue): this function will pull from this queue and
            expect grid ids whose location can be found in the database at
            `database_uri`.
        inference_queue (queue): Planet tiles that need the data inference
            pipeline scheduled will be pushed down this queue.
        database_uri (str): URI to sqlite database.
        planet_api_key (str): key to access Planet's RESTful API.
        mosaic_quad_list_url (str): url that has the Planet global mosaic to
            query for individual tiles.
        planet_quads_dir (str): directory to save downloaded planet tiles in.
            This function will make tree-like subdirectories under the main
            directory based off the last 3 characters of the tile filename.

    Returns:
        None.

    """
    try:
        global FRAGMENT_ID_STATUS_MAP
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
            grid_id = download_queue.get()
            LOGGER.debug('about to fetch grid_id %s', grid_id)
            connection = sqlite3.connect(database_uri, uri=True)
            cursor = connection.cursor()
            cursor.execute(
                'SELECT processing_state, lat_min, lng_min, lat_max, lng_max '
                'FROM grid_status '
                'WHERE grid_id=?', (grid_id,))
            (processing_state, lat_min, lng_min, lat_max, lng_max) = (
                cursor.fetchone())
            # find planet bounding boxes
            LOGGER.debug('fetching %s', (lat_min, lng_min, lat_max, lng_max))
            mosaic_quad_response = get_bounding_box_quads(
                session, mosaic_quad_list_url,
                lng_min, lat_min, lng_max, lat_max)
            mosaic_quad_response_dict = mosaic_quad_response.json()
            # download all the tiles that match
            for mosaic_item in mosaic_quad_response_dict['items']:
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

                fragment_id = '%s_%s' % (grid_id, mosaic_item['id'])
                LOGGER.debug(raster_info)
                with GLOBAL_LOCK:
                    FRAGMENT_ID_STATUS_MAP[fragment_id] = {
                        'bounds':
                            [[raster_wgs84_bb[1], raster_wgs84_bb[0]],
                             [raster_wgs84_bb[3], raster_wgs84_bb[2]]],
                        'color': STATE_TO_COLOR['downloaded']
                    }

                LOGGER.debug('downloaded %s', download_url)
                inference_queue.put((fragment_id, download_raster_path))
            LOGGER.debug(
                '# TODO: update the status to indicate grid is downloaded')
            with GLOBAL_LOCK:
                WORKING_GRID_ID_STATUS_MAP[grid_id] = 'downloaded'
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


def inference_worker(inference_queue, ro_database_uri, tf_model_path):
    """Take large tiles and search for dams.

    Parameters:
        inference_queue (queue): will get ('fragment_id', 'tile_path') tuples
            where 'tile_path' is a path to a geotiff that can be searched
            for dam bounding boxes.
        ro_database_uri (str): URI to read-only version of database that's
            used to search for pre-known dams.

    """
    try:
        LOGGER.debug('database uri: %s model path: %s', ro_database_uri, tf_model_path)
        tf_graph = load_model(tf_model_path)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        while True:
            fragment_id, tile_path = inference_queue.get()
            LOGGER.debug('doing inference on %s', tile_path)
            LOGGER.debug('search for dam bounding boxes in this region')

            raster_info = pygeoprocessing.get_raster_info(tile_path)
            x_size, y_size = raster_info['raster_size']
            for i_off in range(0, x_size, DICE_SIZE[0]):
                for j_off in range(0, y_size, DICE_SIZE[1]):
                    ul_corner = gdal.ApplyGeoTransform(
                        raster_info['geotransform'], i_off, j_off)
                    lr_corner = gdal.ApplyGeoTransform(
                        raster_info['geotransform'],
                        i_off+DICE_SIZE[0], j_off+DICE_SIZE[1])
                    xmin, xmax = sorted([ul_corner[0], lr_corner[0]])
                    ymin, ymax = sorted([ul_corner[1], lr_corner[1]])

                    clipped_raster_path = os.path.join(
                        WORKSPACE_DIR, '%d_%d.tif' % (i_off, j_off))
                    LOGGER.debug("clip to %s", [xmin, ymin, xmax, ymax])
                    pygeoprocessing.warp_raster(
                        tile_path, raster_info['pixel_size'],
                        clipped_raster_path, 'near',
                        target_bb=[xmin, ymin, xmax, ymax])
                    # TODO: get the list of bb coords here and then make
                    # a transformed lat/lng bounding box and also save the
                    # image file somewhere and also make an entry in the
                    # database
                    detection_result = do_detection(
                        tf_graph, THRESHOLD_LEVEL, clipped_raster_path)
                    if detection_result:
                        image_bb_path, coord_list = detection_result
                        for coord_tuple in coord_list:
                            source_srs = osr.SpatialReference()
                            source_srs.ImportFromWkt(raster_info['projection'])
                            to_wgs84 = osr.CoordinateTransformation(
                                source_srs, wgs84_srs)
                            ul_point = ogr.Geometry(ogr.wkbPoint)
                            LOGGER.debug("*********** coord_tuple: %s", coord_tuple)
                            ul_point.AddPoint(coord_tuple[0][0], coord_tuple[0][1])
                            ul_point.Transform(to_wgs84)
                            lr_point = ogr.Geometry(ogr.wkbPoint)
                            lr_point.AddPoint(coord_tuple[1][0], coord_tuple[1][1])
                            lr_point.Transform(to_wgs84)
                            LOGGER.debug(
                                'found dams at %s %s image %s',
                                str((ul_point.GetX(), ul_point.GetY())),
                                str((lr_point.GetX(), lr_point.GetY())),
                                image_bb_path)

            # TODO: write code here to highlight where a dam was found
            # with GLOBAL_LOCK:
            #     for (dam_id, lat_min, lng_min, lat_max, lng_max) in (
            #             cursor.fetchall()):
            #         fragment_dam_id = '%s_%s' % (fragment_id, dam_id)
            #         IDENTIFIED_DAMS[fragment_dam_id] = {
            #             'color': STATE_TO_COLOR['complete'],
            #             'bounds': [
            #                 [lat_min, lng_min],
            #                 [lat_max, lng_max]],
            #         }
            #         LOGGER.debug("FOUND A DAM AT %s", raster_wgs84_bb)

            with GLOBAL_LOCK:
                FRAGMENT_ID_STATUS_MAP[fragment_id]['color'] = (
                    STATE_TO_COLOR['analyzing'])
    except Exception:
        LOGGER.exception("Exception in inference_worker")


def do_detection(detection_graph, threshold_level, image_path):
    """Detect whatever the graph is supposed to detect on a single image.

    Parameters:
        detection_graph (tensorflow Graph): a loaded graph that can accept
            images of the size in `image_path`.
        threshold_level (float): the confidence threshold level to cut off
            classification
        image_path (str): path to an image that `detection_graph` can parse.

    Returns:
        None.

    """
    base_array = gdal.Open(image_path).ReadAsArray().astype(numpy.uint8)
    image_array = numpy.dstack(
        [base_array[0, :, :],
         base_array[1, :, :],
         base_array[2, :, :]])
    LOGGER.debug('detection on %s', image_path)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_array_expanded = numpy.expand_dims(image_array, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            box = detection_graph.get_tensor_by_name('detection_boxes:0')
            score = detection_graph.get_tensor_by_name('detection_scores:0')
            clss = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (box_list, score_list, _, _) = sess.run(
                [box, score, clss, num_detections],
                feed_dict={image_tensor: image_array_expanded})

    # draw a bounding box
    local_box_list = []
    for box, score in zip(box_list[0], score_list[0]):
        LOGGER.debug((box, score))
        if score < threshold_level:
            break
        coords = (
            (box[1] * image_array.shape[1],
             box[0] * image_array.shape[0]),
            (box[3] * image_array.shape[1],
             box[2] * image_array.shape[0]))
        local_box = shapely.geometry.box(
            min(coords[1], coords[3]), min(coords[0], coords[2]),
            max(coords[1], coords[3]), max(coords[0], coords[2]))

        local_box_list.append(local_box)

    # this joins any overlapping-bounding boxes together
    bb_box_list = []
    while local_box_list:
        local_box = local_box_list.pop()
        tmp_box_list = []
        n_intersections = 0
        while local_box_list:
            test_box = local_box_list.pop()
            if local_box.intersects(test_box):
                local_box = local_box.intersection(test_box)
                n_intersections += 1
            else:
                tmp_box_list.append(test_box)
        local_box_list = tmp_box_list
        if n_intersections > 0:
            box_list.append(local_box)

    if bb_box_list:
        lat_lng_list = []
        image = PIL.Image.fromarray(image_array).convert("RGB")
        image_draw = PIL.ImageDraw.Draw(image)
        geotransform = pygeoprocessing.get_raster_info(
            image_path)['geotransform']
        for box in bb_box_list:
            image_draw.rectangle(coords, outline='RED')
            ul_corner = gdal.ApplyGeoTransform(
                geotransform, float(box.bounds[0]), float(box.bounds[1]))
            lr_corner = gdal.ApplyGeoTransform(
                geotransform, float(box.bounds[2]), float(box.bounds[3]))
            lat_lng_list.append((ul_corner, lr_corner))
        del image_draw
        image_path = '%s.png' % os.path.splitext(image_path)[0]
        image.save(image_path)
        return image_path, lat_lng_list
    else:
        return None



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

    inference_model_path = os.path.join(
        WORKSPACE_DIR, os.path.basename(INFERENCE_MODEL_URL))
    task_graph.add_task(
        func=ecoshard.download_url,
        args=(INFERENCE_MODEL_URL, inference_model_path),
        target_path_list=[inference_model_path],
        task_name='download TF model')
    task_graph.join()
    ro_database_uri = 'file:%s?mode=ro' % DATABASE_PATH

    download_work_queue = queue.Queue(2)
    inference_queue = queue.Queue()

    schedule_worker_thread = threading.Thread(
        target=schedule_worker,
        args=(download_work_queue, ro_database_uri))
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
            download_work_queue, inference_queue, ro_database_uri,
            planet_api_key, mosaic_quad_list_url, planet_quads_dir))
    download_worker_thread.start()

    inference_worker_thread = threading.Thread(
        target=inference_worker,
        args=(inference_queue, ro_database_uri, inference_model_path))
    inference_worker_thread.start()

    APP.run(host='0.0.0.0', port=80)
    task_graph.close()
    schedule_worker_thread.join()


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    return app


if __name__ == '__main__':
    GLOBAL_LOCK = threading.Lock()
    WORKING_GRID_ID_STATUS_MAP = {}
    FRAGMENT_ID_STATUS_MAP = {}
    IDENTIFIED_DAMS = {}
    SESSION_UUID = uuid.uuid4().hex
    main()
