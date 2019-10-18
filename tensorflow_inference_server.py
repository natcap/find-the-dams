"""RESTful server to process an image.

Run like this:

    docker run -it --rm -p 8080:8080 {--runtime=nvidia} \
        natcap/dam-inference-server-cpu{gpu}:0.0.1 \
        "python tensorflow_inference_server.py 8080"

"""
import logging
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid

from flask import Flask
import flask
import numpy
import PIL
import PIL.ImageDraw
import shapely.geometry
import tensorflow as tf


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
TF_GRAPH_PATH = sys.argv[1]


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
APP_PORT = int(sys.argv[2])

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
    LOGGER.debug('detection on %s (%s)', png_path, str(image_array.shape))
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


if __name__ == '__main__':
    print(APP.root_path)
    SESSION_MANAGER_LOCK = threading.Lock()
    WORK_QUEUE = queue.Queue()
    inference_thread = threading.Thread(
        target=inference_worker,
        args=(TF_GRAPH_PATH, WORK_QUEUE))
    inference_thread.start()
    garbage_collection_thread = threading.Thread(target=garbage_collection)
    garbage_collection_thread.start()

    APP.run(host='0.0.0.0', port=APP_PORT)
