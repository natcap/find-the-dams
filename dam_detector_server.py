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
import pathlib
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


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.INFO)

WORLD_BORDERS_VECTOR_PATH = 'world_borders.gpkg'
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace'
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'natgeo_dams_database_2020_07_01.db')
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
            "lng_max FROM work_status")

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
            "FROM detected_dams "
            "WHERE dam_id_n>? "
            "ORDER BY dam_id_n "
            "LIMIT %d" % QUERY_LIMIT, (last_known_dam_id,))
        dam_count = 0
        max_dam_id = last_known_dam_id
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


@retrying.retry(wait_exponential_multiplier=100, wait_exponential_max=1000)
def _execute_sqlite(
        sqlite_command, database_path,
        argument_list=None, mode='read_only', execute='execute', fetch=None):
    """Execute SQLite command and attempt retries on a failure.

    Args:
        sqlite_command (str): a well formatted SQLite command.
        database_path (str): path to the SQLite database to operate on.
        argument_list (list): `execute == 'execute` then this list is passed to
            the internal sqlite3 `execute` call.
        execute (str): must be either 'execute', 'many', or 'script'.
        fetch (str): if not `None` can be either 'all' or 'one'.
            If not None the result of a fetch will be returned by this
            function.

    Returns:
        result of fetch if `fetch` is not None.

    """
    cursor = None
    connection = None
    if argument_list is None:
        argument_list = []
    try:
        if fetch:
            ro_uri = r'%s?mode=ro' % pathlib.Path(
                os.path.abspath(database_path)).as_uri()
            LOGGER.debug(
                '%s exists: %s', ro_uri, os.path.exists(os.path.abspath(
                    database_path)))
            connection = sqlite3.connect(ro_uri, uri=True)
        else:
            connection = sqlite3.connect(database_path)

        if execute == 'execute':
            cursor = connection.execute(sqlite_command, argument_list)
        elif execute == 'many':
            cursor = connection.executemany(sqlite_command, argument_list)
        elif execute == 'script':
            cursor = connection.executescript(sqlite_command)
        else:
            raise ValueError('Unknown execute mode: %s' % execute)

        result = None
        payload = None
        if fetch == 'all':
            payload = (cursor.fetchall())
        elif fetch == 'one':
            payload = (cursor.fetchone())
        elif fetch is not None:
            raise ValueError('Unknown fetch mode: %s' % fetch)
        if payload is not None:
            result = list(payload)
        cursor.close()
        connection.commit()
        connection.close()
        return result
    except Exception:
        LOGGER.exception('Exception on _execute_sqlite: %s', sqlite_command)
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.commit()
            connection.close()
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


def main():
    """Entry point."""
    for dirname in [WORKSPACE_DIR,]:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    unprocessed_quads = _execute_sqlite(
        '''
        SELECT grid_id, lng_min, lat_min, lng_max, lat_max
        FROM work_status
        WHERE processed=0
        ''', DATABASE_PATH, fetch='all')

    LOGGER.debug(unprocessed_quads)
    return

    # TODO: get unprocessed quads
    #   TODO: send unprocessed quads to work queue
    # TODO: monitor for new or dead hosts
    #   TODO: manage new hosts with a new thread

    APP.run(host='0.0.0.0', port=80)


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
    args = parser.parse_args()

    GLOBAL_LOCK = threading.Lock()
    GLOBAL_HOST_SET = set()
    GLOBAL_HOST_QUEUE = set()
    WORKING_GRID_ID_STATUS_MAP = {}
    FRAGMENT_ID_STATUS_MAP = {}
    SESSION_UUID = uuid.uuid4().hex
    main()
