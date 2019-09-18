"""Script to classify a single image."""
import json
import sys
import logging
import os
import threading
import ast

import pygeoprocessing
import shutil
import retrying
import requests
import taskgraph
import PIL
import PIL.ImageDraw
import tensorflow as tf
import numpy
from flask import Flask
import flask
from osgeo import gdal

MODEL_PATH = 'models/fasterRCNN_08-26-withnotadams.pb'
#MODEL_PATH = 'models/fasterRCNN_07-27_newimagery.pb'
IMAGE_PATH = 'test_data/11133.png'
#IMAGE_PATH = 'training_data/1005.png'
#IMAGE_PATH = 'big_image.png'
#IMAGE_PATH = 'test_a.png'
#IMAGE_PATH = 'test_b.png'
#IMAGE_PATH = 'small_test.png'
#IMAGE_PATH = 'so_small.png'
WORKSPACE_DIR = 'workspace'
PLANET_API_KEY_FILE = 'planet_api_key.txt'
ACTIVE_MOSAIC_JSON_PATH = os.path.join(WORKSPACE_DIR, 'active_mosaic.json')
LOGGING_LEVEL = logging.DEBUG
REQUEST_TIMEOUT = 5.0
PROCESSING_REQUEST = False
ACTIVE_QUAD = None
MOSAIC_QUAD_LIST_URL = None
THRESHOLD_LEVEL = 0.2
DICE_SIZE = (419, 419)

APP = Flask(
    __name__, static_url_path='', static_folder='')

logging.basicConfig(
    level=LOGGING_LEVEL,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


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
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


@APP.route('/process_quad/', methods=['POST'])
def process_quad():
    """Return results about polygons that are processing."""
    try:
        global PROCESSING_REQUEST
        global ACTIVE_QUAD
        with APP_LOCK:
            if PROCESSING_REQUEST:
                return (
                    'Busy, already processing quad %s' % str(ACTIVE_QUAD), 403)
            request_json = flask.request.get_json()
            quad_payload = ast.literal_eval(request_json['quad'])
            LOGGER.debug(request_json)
            LOGGER.debug(quad_payload)
            PROCESSING_REQUEST = True

            # just make sure it's really xmin, ymin, xmax, ymax,
            lat_min, lat_max = sorted([quad_payload[0], quad_payload[2]])
            lng_min, lng_max = sorted([quad_payload[1], quad_payload[3]])
            ACTIVE_QUAD = (lat_min, lng_min, lat_max, lng_max)

            quad_worker_thread = threading.Thread(
                target=quad_worker, args=ACTIVE_QUAD)
            quad_worker_thread.start()
            return str(ACTIVE_QUAD)
    except Exception as e:
        LOGGER.exception('something bad happened')
        return e.msg, 500


def do_detection(detection_graph, image_path):
    """Detect whatever the graph is supposed to detect on a single image.

    Parameters:
        detection_graph (tensorflow Graph): a loaded graph that can accept
            images of the size in `image_path`.
        image_path (str): path to an image that `detection_graph` can parse.

    Returns:
        None.

    """
    image = PIL.Image.open(image_path).convert("RGB")
    image.show()
    image_array = numpy.array(image)

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
            image_draw = PIL.ImageDraw.Draw(image)
            for box, score in zip(box_list[0], score_list[0]):
                LOGGER.debug((box, score))
                if score <= 0.0:
                    break
                coords = (
                    (box[1] * image_array.shape[1],
                     box[0] * image_array.shape[0]),
                    (box[3] * image_array.shape[1],
                     box[2] * image_array.shape[0]))
                LOGGER.debug((box, coords, image_array.shape))
                image_draw.rectangle(coords, outline='RED')
            del image_draw
            image.save('bb.png')


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
        LOGGER.exception(f"download of {url} to {target_file_path} failed")
        raise


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


def quad_worker(lat_min, lng_min, lat_max, lng_max):
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
    LOGGER.debug('starting fetch queue worker')
    session = requests.Session()
    session.auth = (planet_api_key, '')
    quad_worker_task_graph = taskgraph.TaskGraph(
        os.path.join(WORKSPACE_DIR, 'quad_worker_taskgraph'), -1)

    LOGGER.debug('fetching %s', (lat_min, lng_min, lat_max, lng_max))
    mosaic_quad_response = get_bounding_box_quads(
        session, MOSAIC_QUAD_LIST_URL,
        lng_min, lat_min, lng_max, lat_max)
    mosaic_quad_response_dict = mosaic_quad_response.json()
    # download all the tiles that match
    for mosaic_item in mosaic_quad_response_dict['items']:
        download_url = (mosaic_item['_links']['download'])
        suffix_subdir = os.path.join(
            *reversed(mosaic_item["id"][-4::]))
        download_raster_path = os.path.join(
            planet_quads_dir, suffix_subdir,
            f'{mosaic_item["id"]}.tif')
        download_task = quad_worker_task_graph.add_task(
            func=download_url_to_file,
            args=(download_url, download_raster_path),
            target_path_list=[download_raster_path],
            task_name='download %s' % os.path.basename(
                download_raster_path))
        _ = quad_worker_task_graph.add_task(
            func=inference_on_quad,
            args=(
                download_raster_path, MODEL_PATH, THRESHOLD_LEVEL, DICE_SIZE,
                WORKSPACE_DIR),
            dependent_task_list=[download_task],
            task_name='inference on quad %s' % os.path.basename(
                download_raster_path))
    quad_worker_task_graph.join()

    global PROCESSING_REQUEST
    global ACTIVE_QUAD
    with APP_LOCK:
        PROCESSING_REQUEST = False
        ACTIVE_QUAD = None


def inference_on_quad(
        quad_raster_path, frozen_tf_path, threshold_level, dice_size,
        workspace_dir):
    """Run TensorFlow inference on a given quad.

    Parameters:
        quad_raster_path (str): path to a GeoTIFF to search for objects
            recognized by `frozen_tf_path`.
        frozen_tf_path (str): path to a frozen TensorFlow graph used for
            object detection.
        threshold_level (float): a number between 0..1 indicating what
            "confidence level" from the TF graph to set as a recognized
            object.
        dice_size (tuple): a (width, height) pixel size to cut the
            quad into for searching.
        workspace_dir (str): a directory that can be used for intermediate
            files.

    Returns:
        None.

    """
    tf_graph = load_model(frozen_tf_path)
    try:
        os.makedirs(workspace_dir)
    except OSError:
        pass

    raster_info = pygeoprocessing.get_raster_info(quad_raster_path)
    x_size, y_size = raster_info['raster_size']
    for i_off in range(0, x_size, dice_size[0]):
        for j_off in range(0, y_size, dice_size[1]):
            ul_corner = gdal.ApplyGeoTransform(
                raster_info['geotransform'], i_off, j_off)
            lr_corner = gdal.ApplyGeoTransform(
                raster_info['geotransform'],
                i_off+dice_size[0], j_off+dice_size[1])
            xmin, xmax = sorted([ul_corner[0], lr_corner[0]])
            ymin, ymax = sorted([ul_corner[1], lr_corner[1]])

            clipped_raster_path = os.path.join(
                WORKSPACE_DIR, '%d_%d.tif' % (i_off, j_off))
            LOGGER.debug("clip to %s", [xmin, ymin, xmax, ymax])
            pygeoprocessing.warp_raster(
                quad_raster_path, raster_info['pixel_size'],
                clipped_raster_path, 'near',
                target_bb=[xmin, ymin, xmax, ymax])
            #do_detection(tf_graph, clipped_raster_path)


if __name__ == '__main__':
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
        'using this mosaic: '
        f"""{active_mosaic['last_acquired']} {active_mosaic['interval']} {
            active_mosaic['grid']['resolution']}""")

    MOSAIC_QUAD_LIST_URL = (
        f"""https://api.planet.com/basemaps/v1/mosaics/"""
        f"""{active_mosaic['id']}/quads""")

    APP_LOCK = threading.Lock()
    APP.run(host='0.0.0.0', port=8080)

