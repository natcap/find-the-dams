"""RESTful server to process an image.

Run like this:

    docker run -it --rm -p 8080:8080 {--runtime=nvidia} \
        natcap/dam-inference-server-cpu{gpu}:0.0.1 \
        "python tensorflow_inference_server.py 8080"

"""
import argparse
import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid

from flask import Flask
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import cv2
import flask
import numpy
import PIL
import PIL.ImageDraw
import pygeoprocessing
import requests
import retrying
import shapely.geometry

from keras_retinanet import models
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
import keras

WORKSPACE_DIR = 'workspace_tf_server'
ANNOTATED_IMAGE_DIR = os.path.join(WORKSPACE_DIR, 'annotated_images')
for dirname in [WORKSPACE_DIR, ANNOTATED_IMAGE_DIR]:
    try:
        os.makedirs(dirname)
    except OSError:
        pass

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

APP = Flask(__name__, static_url_path='', static_folder='')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'od_workspace')
THRESHOLD_LEVEL = 0.08

try:
    shutil.rmtree(UPLOAD_FOLDER)
except OSError:
    pass
try:
    os.makedirs(UPLOAD_FOLDER)
except OSError:
    pass

# Configure Flask app and the logo upload folder
APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# map session ids to current state
# can be
#  * 'waiting for upload'
#  * 'processing'
#  * 'complete'
#       * if complete then has 'annotated_png_url_base' to turn into url
#  * '<traceback>' with 500 error
SESSION_MANAGER_MAP = {}

# this is used to clear out unused sessions, it will have a
# (last accessed timestamp, files created list) tuple that a thread monitors
# and cleans periodically.
LAST_ACCESSED_SESSION_MAP = {}
# if session not accessed within this amount of time files in
# LAST_ACCESSED_SESSION_MAP will be removed
CLEANUP_WAIT_TIME = 60.0


@APP.route('/api/v1/detect_dam', methods=['POST'])
def detect_dam_init():
    """Initialize a new dam classifying image."""
    session_id = str(uuid.uuid4())
    print(session_id)
    with SESSION_MANAGER_LOCK:
        SESSION_MANAGER_MAP[session_id] = {
            'status': 'waiting for upload',
            'http_status_code': 200,
        }
    return {
        'upload_url': flask.url_for(
            'detect_dam', _external=True, session_id=session_id)
    }


@APP.route('/api/v1/detect_dam/<string:session_id>', methods=['PUT'])
def detect_dam(session_id):
    """Flask entry point."""
    with SESSION_MANAGER_LOCK:
        if session_id not in SESSION_MANAGER_MAP:
            return ('%s not a valid session', 400)
        session_state = SESSION_MANAGER_MAP[session_id]
    if session_state['status'] != 'waiting for upload':
        return ('file already uploaded', session_state['status'], 400)
    target_path = os.path.join(UPLOAD_FOLDER, '%s.png' % session_id)
    flask.request.files['file'].save(target_path)
    WORK_QUEUE.put((session_id, target_path))
    with SESSION_MANAGER_LOCK:
        LAST_ACCESSED_SESSION_MAP[session_id] = {
            'last_time': time.time(),
            'file_list': [target_path],
        }
        SESSION_MANAGER_MAP[session_id] = {
            'status': 'processing',
            'status_url': flask.url_for(
                'get_status', _external=True, session_id=session_id),
            'http_status_code': 200,
        }
        return session_map_to_response(SESSION_MANAGER_MAP[session_id])


@APP.route('/api/v1/get_status/<string:session_id>', methods=['GET'])
def get_status(session_id):
    """Return status of processing."""
    with SESSION_MANAGER_LOCK:
        if session_id not in SESSION_MANAGER_MAP:
            return "%s not found" % session_id, 400
        LAST_ACCESSED_SESSION_MAP[session_id]['last_time'] = time.time()
        current_session = SESSION_MANAGER_MAP[session_id]
        if 'annotated_png_filename' in current_session:
            current_session['annotated_png_url'] = (
                flask.url_for(
                    'download_result', _external=True,
                    filename=(
                        current_session['annotated_png_filename'])))
            LOGGER.debug('rewriting url %s', current_session)
        return session_map_to_response(current_session)


@APP.route('/api/v1/download/<string:filename>', methods=['GET'])
def download_result(filename):
    """Download a result if possible."""
    local_path = os.path.join(ANNOTATED_IMAGE_DIR, filename)
    LOGGER.debug(
        'getting file: %s %s', local_path, os.path.exists(local_path))
    return flask.send_from_directory('.', local_path)


def do_detection(tf_graph, threshold_level, png_path):
    """Detect whatever the graph is supposed to detect on a single image.

    Parameters:
        tf_graph (tensorflow Graph): a loaded graph that can accept
            images.
        threshold_level (float): the confidence threshold level to cut off
            classification
        png_path (str): path to an image that `tf_graph` can parse.


    Returns:
        None.

    """
    image = PIL.Image.open(png_path).convert("RGB")
    image_array = numpy.array(image.getdata()).reshape(
        image.size[1], image.size[0], 3)
    LOGGER.info('detection on %s (%s)', png_path, str(image_array.shape))
    with tf_graph.as_default():
        with tf.Session(graph=tf_graph) as sess:
            image_array_expanded = numpy.expand_dims(image_array, axis=0)
            image_tensor = tf_graph.get_tensor_by_name('image_tensor:0')
            box = tf_graph.get_tensor_by_name('detection_boxes:0')
            score = tf_graph.get_tensor_by_name('detection_scores:0')
            clss = tf_graph.get_tensor_by_name('detection_classes:0')
            num_detections = tf_graph.get_tensor_by_name('num_detections:0')

            (box_list, score_list, _, _) = sess.run(
                [box, score, clss, num_detections],
                feed_dict={image_tensor: image_array_expanded})

    # draw a bounding box
    local_box_list = []
    for box, score in reversed(sorted(
            zip(box_list[0], score_list[0]), key=lambda x: x[1])):
        if score < threshold_level:
            break

        # make sure in bounds
        if ((box[1] < .1 and box[3] < .1) or
                (box[0] < .1 and box[2] < .1) or
                (box[1] > .9 and box[3] > .9) or
                (box[0] > .9 and box[2] > .9) or
                (abs(box[0] - box[2]) >= .9) or
                (abs(box[1] - box[3]) >= .9)):
            # this is an edge box -- probably not a dam and is an artifact of
            # fasterRCNN_08-26-withnotadams_md5_83f58894e34e1e785fcaa2dbc1d3ec7a.pb
            continue
        coords = (
            (box[1] * image.size[1],
             box[0] * image.size[0]),
            (box[3] * image.size[1],
             box[2] * image.size[0]))
        local_box = shapely.geometry.box(
            min(coords[0][0], coords[1][0]),
            min(coords[0][1], coords[1][1]),
            max(coords[0][0], coords[1][0]),
            max(coords[0][1], coords[1][1]))
        local_box_list.append(local_box)
    LOGGER.debug('found %d bounding boxes', len(local_box_list))

    # this unions any overlapping-bounding boxes together
    bb_box_list = []
    while local_box_list:
        local_box = local_box_list.pop()
        tmp_box_list = []
        n_intersections = 0
        while local_box_list:
            test_box = local_box_list.pop()
            if local_box.intersects(test_box):
                local_box = local_box.union(test_box)
                n_intersections += 1
            else:
                tmp_box_list.append(test_box)
        local_box_list = tmp_box_list
        bb_box_list.append(local_box)

    if bb_box_list:
        LOGGER.debug('*** found a bounding box')

    return bb_box_list, png_path


def session_map_to_response(session_map):
    """Return HTTP response code if it's in there."""
    if 'http_status_code' in session_map:
        return (session_map, session_map['http_status_code'])
    return session_map


def inference_worker(
        tf_graph_path, work_queue, inference_worker_payload_queue):
    """Process images as they are sent to the server.

    Processes tuples from work_queue and updates SESSION_MANAGER_MAP either
    with a dict of bounding box and annotated image results or an error
    message from an exception.

    Parameters:
        tf_graph_path (str): path to Tensorflow 1.12.0 frozen inference graph.
        work_queue (queue): expect (session_id, png_path) tuples
        inference_worker_payload_queue (queue): pass resultof do detection
            for finalization here.

    Returns:
        None

    """
    tf_graph = load_model(TF_GRAPH_PATH)

    while True:
        try:
            session_id, png_path = work_queue.get()
            LOGGER.debug('processing %s', png_path)
            dam_image_workspace = os.path.join(WORKSPACE_DIR, session_id)
            try:
                os.makedirs(dam_image_workspace)
            except OSError:
                pass
            try:
                payload = do_detection(tf_graph, THRESHOLD_LEVEL, png_path)
                inference_worker_payload_queue.put((payload, session_id))
            except Exception:
                with SESSION_MANAGER_LOCK:
                    SESSION_MANAGER_MAP[session_id] = {
                        'status': traceback.format_exc(),
                        'http_status_code': 500,
                    }

        except Exception:
            LOGGER.exception('exception in inference worker')


def render_image_and_finalize(inference_worker_payload_queue):
    """Take a payload and render the image and update the session manager."""
    while True:
        try:
            (bb_box_list, png_path), session_id = (
                inference_worker_payload_queue.get())
            image = PIL.Image.open(png_path).convert("RGB")
            bb_list = []
            image_draw = PIL.ImageDraw.Draw(image)
            for box in bb_box_list:
                image_draw.rectangle(box.bounds, outline='RED')
                ul_corner = (float(box.bounds[0]), float(box.bounds[1]))
                lr_corner = (float(box.bounds[2]), float(box.bounds[3]))
                bb_list.append((ul_corner, lr_corner))
            del image_draw
            annotated_path = os.path.join(
                ANNOTATED_IMAGE_DIR,
                '%s_annotated%s' % os.path.splitext(
                    os.path.basename(png_path)))
            image.save(annotated_path)
            LOGGER.debug('saved to %s', annotated_path)

            with SESSION_MANAGER_LOCK:
                SESSION_MANAGER_MAP[session_id] = {
                    'status': 'complete',
                    'annotated_png_filename': os.path.basename(
                        annotated_path),
                    'bounding_box_list': bb_list,
                    'http_status_code': 200,
                }
                LAST_ACCESSED_SESSION_MAP[session_id]['last_time'] = (
                    time.time())
                LAST_ACCESSED_SESSION_MAP[session_id]['file_list'].append(
                    annotated_path)
        except Exception:
            LOGGER.exception('exception in render_image_and_finalize')


def garbage_collection():
    """Periodically checks if LAST_ACESS_SESSION_MAP needs culling."""
    while True:
        try:
            time.sleep(CLEANUP_WAIT_TIME)
            current_time = time.time()
            LOGGER.debug('garbage collecting, current time %s', current_time)
            with SESSION_MANAGER_LOCK:
                session_remove_list = []
                for session_id, access_map in \
                        LAST_ACCESSED_SESSION_MAP.items():
                    if (current_time > access_map['last_time'] +
                            CLEANUP_WAIT_TIME):
                        for file_path in access_map['file_list']:
                            LOGGER.debug(
                                'removing %s after %.2f seconds', file_path,
                                current_time - access_map['last_time'])
                            os.remove(file_path)
                        session_remove_list.append(session_id)
                for session_id in session_remove_list:
                    del LAST_ACCESSED_SESSION_MAP[session_id]
                    del SESSION_MANAGER_MAP[session_id]
        except Exception:
            LOGGER.exception('exception in `garbage_collection')


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


def old_inference_worker(
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
                        tf_graph, THRESHOLD_LEVEL, clipped_raster_path)
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


# WORK_STATUS maps quad url to a 'idle', 'working', 'error', or list result
# if complete
QUAD_URL_TO_STATUS_MAP = dict()
URL_TO_PROCESS_LIST = []
QUAD_AVAILBLE_EVENT = threading.Event()


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """Compute an image scale size to constrained to min_side/max_side.

    Args:
        min_side (int): The image's min side will be equal to min_side after
            resizing.
        max_side (int): If after resizing the image's max side is above
            max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape
    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def do_inference_worker(model):
    """Calculate inference on the next available URL."""
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)

    while True:
        QUAD_AVAILBLE_EVENT.wait(5.0)
        if not URL_TO_PROCESS_LIST:
            continue
        quad_url = URL_TO_PROCESS_LIST.pop()
        QUAD_URL_TO_STATUS_MAP[quad_url] = 'processing'
        LOGGER.info('downloading ' + quad_url)
        quad_workspace = os.path.join(
            '/', 'data', os.path.basename(os.path.splitext(quad_url)[0]))
        quad_raster_path = os.path.join(
            quad_workspace, os.path.basename(quad_url))
        LOGGER.info('download ' + quad_url)
        with requests.get(quad_url, stream=True, timeout=5.0) as r:
            with open(quad_raster_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        LOGGER.info('process cuts of quad ' + quad_raster_path)
        #raster_info = pygeoprocessing.get_raster_info(quad_raster_path)

        try:
            raw_image = numpy.asarray(PIL.Image.open(
                quad_raster_path).convert('RGB'))[:, :, ::-1].copy()
            image = (
                raw_image.astype(numpy.float32) - [103.939, 116.779, 123.68])
            scale = compute_resize_scale(
                image.shape, min_side=800, max_side=1333)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            if keras.backend.image_data_format() == 'channels_first':
                image = image.transpose((2, 0, 1))

            LOGGER.debug('run inference on image %s', str(image.shape))
            result = model.predict_on_batch(
                numpy.expand_dims(image, axis=0))
            # correct boxes for image scale
            LOGGER.debug('inference complete')
            boxes, scores, labels = result
            boxes /= scale

            # convert box to a list from a numpy array and score to a value
            # from a single element array
            non_max_supression_box_list = []
            box_score_tuple_list = [
                (list(box), score) for box, score in zip(boxes[0], scores[0])
                if score > 0.3]
            while box_score_tuple_list:
                box, score = box_score_tuple_list.pop()
                shapely_box = shapely.geometry.box(*box)
                keep = True
                # this list makes a copy
                for test_box, test_score in list(box_score_tuple_list):
                    shapely_test_box = shapely.geometry.box(*test_box)
                    if shapely_test_box.intersects(shapely_box):
                        if test_score > score:
                            # keep the new one
                            keep = False
                            break
                if keep:
                    non_max_supression_box_list.append((
                        [float(x) for x in box], float(score)))

            LOGGER.debug(non_max_supression_box_list)
        except Exception as e:
            LOGGER.exception('error on processing image')
            return str(e), 500
        # TODO: do inference on all the pieces
        # TODO: store the result in QUAD_URL_TO_STATUS_MAP
        # TODO: delete the quad
        if len(URL_TO_PROCESS_LIST) == 0:
            QUAD_AVAILBLE_EVENT.clear()


@APP.route('/do_inference', methods=['POST'])
def do_inference():
    """Run dam inference on the posted quad."""
    LOGGER.debug(flask.request.data)
    LOGGER.debug(flask.request.json)
    quad_url = flask.request.json['quad_url']

    if quad_url in QUAD_URL_TO_STATUS_MAP:
        return quad_url + ' already scheduled', 500
    QUAD_URL_TO_STATUS_MAP[quad_url] = 'scheduled'
    URL_TO_PROCESS_LIST.append(quad_url)
    QUAD_AVAILBLE_EVENT.set()
    return quad_url + ' is scheduled'


@APP.route('/job_status', methods=['POST'])
def job_status():
    """Report status of given job."""
    # 'idle'
    # 'processing'
    # 'complete'
    # 'error'
    quad_url = flask.request.json['quad_url']
    LOGGER.info('fetch status of ' + quad_url)
    status = QUAD_URL_TO_STATUS_MAP[quad_url]
    QUAD_URL_TO_STATUS_MAP[quad_url] = status
    if not isinstance(status, list):
        return {'status': status}
    else:
        return {'status': 'complete'}


@APP.route('/get_result', methods=['POST'])
def get_result():
    """Get the result for a given quad."""
    delivered = False
    try:
        quad_url = flask.request.json['quad_url']
        LOGGER.info('get result for ' + quad_url)
        status = QUAD_URL_TO_STATUS_MAP[quad_url]
        if isinstance(status, list):
            delivered = True
            return {
                'quad_url': quad_url,
                'dam_bounding_box_list': status
                }
        else:
            return quad_url + ' not complete with status: ' + status, 500
    except Exception as e:
        LOGGER.exception('error on get result for ' + quad_url)
        return str(e), 500
    finally:
        if delivered:
            del QUAD_URL_TO_STATUS_MAP[quad_url]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Carbon edge model')
    parser.add_argument('tensorflow_model_path', help='path to frozen model')
    parser.add_argument(
        '--app_port', default=80, help='server port')
    args = parser.parse_args()

    check_keras_version()
    check_tf_version()

    LOGGER.info(f'loading {args.tensorflow_model_path}')
    model = models.load_model(args.tensorflow_model_path, backbone_name='resnet50')

    do_inference_worker_thread = threading.Thread(
        target=do_inference_worker,
        args=(model,))
    do_inference_worker_thread.daemon = True
    do_inference_worker_thread.start()
    APP.run(host='0.0.0.0', port=args.app_port)

    # SESSION_MANAGER_LOCK = threading.Lock()
    # WORK_QUEUE = queue.Queue()
    # inference_thread = threading.Thread(
    #     target=inference_worker,
    #     args=(TF_GRAPH_PATH, WORK_QUEUE))
    # inference_thread.start()
    # garbage_collection_thread = threading.Thread(target=garbage_collection)
    # garbage_collection_thread.start()
