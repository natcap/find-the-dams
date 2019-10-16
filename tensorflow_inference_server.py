"""RESTful server to process an image."""
import uuid
import shutil
import threading
import os
import logging
import sys
import queue

import shapely.geometry
import PIL
import flask
from flask import Flask
import tensorflow as tf
import numpy


WORKSPACE_DIR = 'workspace_tf_server'
ANNOTATED_IMAGE_DIR = os.path.join(WORKSPACE_DIR, 'annotated_images')
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

APP = Flask(__name__, static_url_path='', static_folder=WORKSPACE_DIR)
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
#  * 'error: <msg>'
#  * <url to download file>
SESSION_MANAGER_MAP = {}


@APP.route('/api/v1/detect_dam', methods=['POST'])
def detect_dam_init():
    """Initialize a new dam classifying image."""
    session_id = str(uuid.uuid4())
    print(session_id)
    with SESSION_MANAGER_LOCK:
        SESSION_MANAGER_MAP[session_id] = {
            'status': 'waiting for upload'
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
        SESSION_MANAGER_MAP[session_id] = {
            'status': 'processing',
            'status_url': flask.url_for(
                'get_status', _external=True, session_id=session_id)
        }
        return SESSION_MANAGER_MAP[session_id]


@APP.route('/api/v1/get_status/<string:session_id>', methods=['GET'])
def get_status(session_id):
    """Returns status of processing."""
    with SESSION_MANAGER_LOCK:
        if session_id not in SESSION_MANAGER_MAP:
            return ('%s not a valid session', 400)
        return SESSION_MANAGER_MAP[session_id]


@APP.route('/api/v1/download/<string:filename>', methods=['POST'])
def download_result(filename):
    """Download a result if possible."""
    return flask.send_from_directory(ANNOTATED_IMAGE_DIR, filename)


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
    image_array = PIL.Image.open(png_path).convert("RGB").getdata()
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
            (box[1] * image_array.shape[1],
             box[0] * image_array.shape[0]),
            (box[3] * image_array.shape[1],
             box[2] * image_array.shape[0]))
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
        bb_list = []
        image = PIL.Image.fromarray(image_array).convert("RGB")
        image_draw = PIL.ImageDraw.Draw(image)
        for box in bb_box_list:
            image_draw.rectangle(coords, outline='RED')
            ul_corner = (float(box.bounds[0]), float(box.bounds[1]))
            lr_corner = (float(box.bounds[2]), float(box.bounds[3]))
            bb_list.append((ul_corner, lr_corner))
        del image_draw
        annotated_path = os.path.join(
            ANNOTATED_IMAGE_DIR,
            '%s_annotated.%s' % os.path.splitext(png_path))
        image.save(annotated_path)
        return annotated_path, bb_list
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


def inference_worker(tf_graph_path, work_queue):
    """Process images as they are sent to the server.

    Processes tuples from work_queue and updates SESSION_MANAGER_MAP either
    with a dict of bounding box and annotated image results or an error
    message from an exception.

    Parameters:
        tf_graph_path (str): path to Tensorflow 1.12.0 frozen inference graph.
        work_queue (queue): expect (session_id, png_path) tuples

    Returns:
        None

    """
    tf_graph = load_model(TF_GRAPH_PATH)

    while True:
        session_id, png_path = work_queue.get()
        LOGGER.debug('processing %s', png_path)
        dam_image_workspace = os.path.join(WORKSPACE_DIR, session_id)
        try:
            os.makedirs(dam_image_workspace)
        except OSError:
            pass
        try:
            payload = do_detection(tf_graph, THRESHOLD_LEVEL, png_path)
        except Exception as e:
            with SESSION_MANAGER_LOCK:
                SESSION_MANAGER_MAP[session_id] = {
                    'status': str(e.msg)
                }

        with SESSION_MANAGER_LOCK:
            annotated_path, bb_list = payload
            SESSION_MANAGER_MAP[session_id] = {
                'status': 'complete',
                'annotated_png_url': flask.url_for(
                    'download_result', _external=True,
                    filename=os.path.basename(annotated_path)),
                'bounding_box_list': bb_list}


if __name__ == '__main__':
    print(APP.root_path)
    SESSION_MANAGER_LOCK = threading.Lock()
    WORK_QUEUE = queue.Queue()
    thread = threading.Thread(
        target=inference_worker,
        args=(TF_GRAPH_PATH, WORK_QUEUE))
    thread.start()
    APP.run(host='0.0.0.0', port=APP_PORT)
