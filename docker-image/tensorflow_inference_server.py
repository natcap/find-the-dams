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
import png
import pygeoprocessing
import requests
import retrying
import shapely.geometry

from keras_retinanet import models
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
import keras

WORKSPACE_DIR = '/usr/local/data_dir'
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

TRAINING_IMAGE_DIMS = (419, 419)

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


def render_bounding_boxes(bb_box_list, png_path):
    """Take a payload and render the image and update the session manager."""
    image = PIL.Image.open(png_path).convert("RGB")
    bb_list = []
    image_draw = PIL.ImageDraw.Draw(image)
    for box in bb_box_list:
        image_draw.rectangle(box, outline='RED')
        ul_corner = (float(box[0]), float(box[1]))
        lr_corner = (float(box[2]), float(box[3]))
        bb_list.append((ul_corner, lr_corner))
    del image_draw
    image.save(png_path)
    LOGGER.debug('saved to %s', png_path)


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


def make_quad_png(
        quad_raster_path, quad_png_path, xoff, yoff, win_xsize, win_ysize):
    """Make a PNG out of a geotiff.

    Parameters:
        quad_raster_path (str): path to target download location.
        quad_png_path (str): path to target png file.
        xoff (int): x offset to read quad array
        yoff (int): y offset to read quad array
        win_xsize (int): size of x window
        win_ysize (int): size of y window

    Returns:
        None.

    """
    raster = gdal.OpenEx(quad_raster_path, gdal.OF_RASTER)
    rgba_array = numpy.array([
        raster.GetRasterBand(i).ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)
        for i in [1, 2, 3, 4]])
    try:
        row_count, col_count = rgba_array.shape[1::]
        image_2d = numpy.transpose(
            rgba_array, axes=[0, 2, 1]).reshape(
            (-1,), order='F').reshape((-1, col_count*4))
        png.from_array(image_2d, 'RGBA').save(quad_png_path)
        return quad_png_path
    except Exception:
        LOGGER.exception(
            'error on %s generate png with array:\n%s\ndims:%s\n'
            'file exists:%s\nxoff=%d, yoff=%d, win_xsize=%d, win_ysize=%d' % (
                quad_raster_path, rgba_array, rgba_array.shape,
                os.path.exists(quad_raster_path),
                xoff, yoff, win_xsize, win_ysize))
        raise


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
        quad_raster_path = os.path.join(
            WORKSPACE_DIR, os.path.basename(quad_url))
        LOGGER.info('download ' + quad_url + ' to ' + quad_raster_path)
        with requests.get(quad_url, stream=True, timeout=5.0) as r:
            with open(quad_raster_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        LOGGER.info('process cuts of quad ' + quad_raster_path)

        quad_info = pygeoprocessing.get_raster_info(quad_raster_path)
        n_cols, n_rows = quad_info['raster_size']
        quad_id = os.path.basename(os.path.splitext(quad_raster_path)[0])
        quad_slice_index = 0
        non_max_supression_box_list = []
        for xoff in range(0, n_cols, TRAINING_IMAGE_DIMS[0]):
            win_xsize = TRAINING_IMAGE_DIMS[0]
            if xoff + win_xsize >= n_cols:
                xoff = n_cols-win_xsize-1
            for yoff in range(0, n_rows, TRAINING_IMAGE_DIMS[1]):
                win_ysize = TRAINING_IMAGE_DIMS[1]
                if yoff + win_ysize >= n_rows:
                    yoff = n_rows-win_ysize-1
                try:
                    quad_png_path = os.path.join(
                        WORKSPACE_DIR, '%s_%d.png' % (
                            quad_id, quad_slice_index))
                    quad_slice_index += 1
                    make_quad_png(
                        quad_raster_path, quad_png_path,
                        xoff, yoff, win_xsize, win_ysize)

                    raw_image = numpy.asarray(PIL.Image.open(
                        quad_png_path).convert('RGB'))[:, :, ::-1].copy()
                    image = (
                        raw_image.astype(numpy.float32) - [
                            103.939, 116.779, 123.68])
                    scale = compute_resize_scale(
                        image.shape, min_side=800, max_side=1333)
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                    if keras.backend.image_data_format() == 'channels_first':
                        image = image.transpose((2, 0, 1))

                    LOGGER.debug('run inference on image %s', quad_png_path)
                    result = model.predict_on_batch(
                        numpy.expand_dims(image, axis=0))
                    # correct boxes for image scale
                    LOGGER.debug('inference complete')
                    LOGGER.debug('results: ' + str(result))
                    boxes, scores, labels = result
                    boxes /= scale

                    # convert box to a list from a numpy array and score to a value
                    # from a single element array
                    box_score_tuple_list = [
                        (list(box), score) for box, score in zip(
                            boxes[0], scores[0]) if score > 0.3]
                    local_box_list = []
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
                            local_box_list.append([float(x) for x in box])
                    non_max_supression_box_list.extend([
                        [bb[0]+xoff, bb[1]+yoff, bb[2]+xoff, bb[3]+yoff]
                        for bb in local_box_list])

                except Exception as e:
                    LOGGER.exception('error on processing image')
                    return str(e), 500
        quad_png_path = '%s.png' % os.path.splitext(quad_raster_path)[0]
        make_quad_png(
            quad_raster_path, quad_png_path, xoff, yoff, None, None)
        render_bounding_boxes(local_box_list, quad_png_path)
        # TODO: store the result in QUAD_URL_TO_STATUS_MAP
        # TODO: delete the quad
        LOGGER.info('done processing quad %s', quad_raster_path)
        LOGGER.debug(non_max_supression_box_list)
        if len(URL_TO_PROCESS_LIST) == 0:
            QUAD_AVAILBLE_EVENT.clear()


@APP.route('/do_inference', methods=['POST'])
def do_inference():
    """Run dam inference on the posted quad."""
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
