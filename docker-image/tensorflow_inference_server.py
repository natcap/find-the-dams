"""RESTful server to process an image.

Run like this:

    docker run -it --rm -p 8080:8080 {--runtime=nvidia} \
        natcap/dam-inference-server-cpu{gpu}:0.0.1 \
        "python tensorflow_inference_server.py 8080"

"""
import argparse
import logging
import os
import queue
import multiprocessing
import subprocess
import sys
import threading
import time

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
import shapely.geometry

from keras_retinanet import models
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
import keras

JOBS_PER_WORKER = 3
WORKSPACE_DIR = '/usr/local/data_dir'
ANNOTATED_IMAGE_DIR = os.path.join(WORKSPACE_DIR, 'annotated_images')
for dirname in [WORKSPACE_DIR, ANNOTATED_IMAGE_DIR]:
    try:
        os.makedirs(dirname)
    except OSError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

APP = Flask(__name__, static_url_path='', static_folder='')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_IMAGE_DIMS = (419, 419)
HEALTHY = True


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
QUAD_URI_TO_STATUS_MAP = dict()
URI_TO_PROCESS_LIST = []
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

    Args:
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
        global HEALTHY
        HEALTHY = False
        LOGGER.exception(
            'error on %s generate png with array:\n%s\ndims:%s\n'
            'file exists:%s\nxoff=%d, yoff=%d, win_xsize=%d, win_ysize=%d' % (
                quad_raster_path, rgba_array, rgba_array.shape,
                os.path.exists(quad_raster_path),
                xoff, yoff, win_xsize, win_ysize))
        raise


def quad_processor(quad_offset_queue, quad_file_path_queue):
    """Process a Quad.

    Args:
        quad_offset_queue (queue): contains
            (quad_png_path, quad_raster_path, xoff, yoff, win_xsize, win_ysize)
            payload for indicating what image to clip.
        quad_file_path_queue (queue): the "output" queue for what file has been
            clipped and any scaling that's done (scale, image).

    Returns:
        never.
    """
    global HEALTHY
    try:
        while True:
            payload = quad_offset_queue.get()
            (quad_png_path, quad_raster_path,
             xoff, yoff, win_xsize, win_ysize) = payload
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
            image = numpy.expand_dims(image, axis=0)
            quad_file_path_queue.put((xoff, yoff, scale, image))
            os.remove(quad_png_path)
    except Exception:
        LOGGER.exception('error occured on quad processor')
        HEALTHY = False
        raise


def do_inference_worker(
        model, quad_offset_queue, quad_file_path_queue, inference_lock):
    """Calculate inference on data coming in on the URI_TO_PROCESS_LIST.

    Other notable global variable is QUAD_AVAILBLE_EVENT that's an event for
    waiting for new work that gets set when new works is recieved.

    Args:
        model (keras model): model used for bounding box prediction
        quad_offset_queue (queue): send to queue for quad processing
        quad_file_path_queue (queue): used for recieving quads that need to be
            inferenced.
        inference_lock (threading.Lock): used to ensure one shot of inference
            goes at a time.

    Returns:
        never
    """
    global HEALTHY
    try:
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        subprocess_result = None
        while True:
            QUAD_AVAILBLE_EVENT.wait(5.0)
            if not URI_TO_PROCESS_LIST:
                continue
            start_time = time.time()
            quad_uri = URI_TO_PROCESS_LIST.pop()
            QUAD_URI_TO_STATUS_MAP[quad_uri] = 'processing'
            quad_raster_path = os.path.join(
                WORKSPACE_DIR, os.path.basename(quad_uri))
            LOGGER.info('download ' + quad_uri + ' to ' + quad_raster_path)
            subprocess_result = subprocess.run(
                '/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp '
                '"%s" %s' % (quad_uri, quad_raster_path), check=True,
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            quad_info = pygeoprocessing.get_raster_info(quad_raster_path)
            n_cols, n_rows = quad_info['raster_size']
            quad_id = os.path.basename(os.path.splitext(quad_raster_path)[0])
            quad_slice_index = 0
            non_max_supression_box_list = []

            LOGGER.info('schedule clip of %s', quad_id)
            for xoff in range(0, n_cols, TRAINING_IMAGE_DIMS[0]):
                win_xsize = TRAINING_IMAGE_DIMS[0]
                if xoff + win_xsize >= n_cols:
                    xoff = n_cols-win_xsize-1
                for yoff in range(0, n_rows, TRAINING_IMAGE_DIMS[1]):
                    win_ysize = TRAINING_IMAGE_DIMS[1]
                    if yoff + win_ysize >= n_rows:
                        yoff = n_rows-win_ysize-1
                    quad_png_path = os.path.join(
                        WORKSPACE_DIR, '%s_%d.png' % (
                            quad_id, quad_slice_index))
                    quad_slice_index += 1
                    quad_offset_queue.put(
                        (quad_png_path, quad_raster_path,
                         xoff, yoff, win_xsize, win_ysize))

            LOGGER.info('schedule inference of %s', quad_id)
            box_score_tuple_list = []
            with inference_lock:
                while quad_slice_index > 0:
                    quad_slice_index -= 1
                    xoff, yoff, scale, image = quad_file_path_queue.get()
                    result = model.predict_on_batch(image)
                    # correct boxes for image scale
                    boxes, scores, labels = result
                    boxes /= scale

                    # convert box to a list from a numpy array and score to a
                    # value from a single element array
                    box_score_tuple_list.extend([
                        ([box[0]+xoff, box[1]+yoff, box[2]+xoff, box[3]+yoff],
                         score) for box, score in zip(
                            boxes[0], scores[0]) if score > 0.3])

            # quad is now processed, it can be removed
            os.remove(quad_raster_path)
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
                    non_max_supression_box_list.append(box)

            #quad_png_path = '%s.png' % os.path.splitext(quad_raster_path)[0]
            # make_quad_png(
            #     quad_raster_path, quad_png_path, 0, 0, None, None)
            # render_bounding_boxes(non_max_supression_box_list, quad_png_path)
            lat_lng_bb_list = []
            for bounding_box in non_max_supression_box_list:
                local_coord_bb = []
                for offset in [0, 2]:
                    coords = list(gdal.ApplyGeoTransform(
                        quad_info['geotransform'],
                        bounding_box[0+offset],
                        bounding_box[1+offset]))
                    local_coord_bb.extend(coords)
                transformed_bb = pygeoprocessing.transform_bounding_box(
                    local_coord_bb, quad_info['projection_wkt'],
                    wgs84_srs.ExportToWkt())
                lat_lng_bb_list.append(transformed_bb)
            QUAD_URI_TO_STATUS_MAP[quad_uri] = lat_lng_bb_list
            LOGGER.info(
                'done processing quad %s took %ss',
                quad_raster_path, str(time.time()-start_time))
            if len(URI_TO_PROCESS_LIST) == 0:
                QUAD_AVAILBLE_EVENT.clear()
    except Exception:
        LOGGER.exception('error occured on inference worker')
        if subprocess_result:
            LOGGER.error(subprocess_result)
        QUAD_URI_TO_STATUS_MAP[quad_uri] = 'error'
        HEALTHY = False
        raise


@APP.route('/health_check', methods=['GET'])
def health_check():
    """Return 200 if healthy, 500 if not."""
    global HEALTHY
    if HEALTHY:
        return 'healthy', 200
    else:
        return 'error', 500


@APP.route('/do_inference', methods=['POST'])
def do_inference():
    """Run dam inference on the posted quad."""
    try:
        LOGGER.debug(flask.request.json)
        quad_uri_list = flask.request.json['quad_uri_list']
        for quad_uri in quad_uri_list:
            if quad_uri not in QUAD_URI_TO_STATUS_MAP:
                QUAD_URI_TO_STATUS_MAP[quad_uri] = 'scheduled'
                URI_TO_PROCESS_LIST.append(quad_uri)
                QUAD_AVAILBLE_EVENT.set()
        return str(quad_uri_list) + ' is scheduled'
    except Exception:
        LOGGER.exception('something went wrong')
        global HEALTHY
        HEALTHY = False
        raise


@APP.route('/job_status', methods=['POST'])
def job_status():
    """Report status of given job."""
    # 'scheduled'
    # 'processing'
    # 'error'
    # [a list of obunding boxes]
    try:
        quad_uri_list = flask.request.json['quad_uri_list']
        status_list = []
        for quad_uri in quad_uri_list:
            LOGGER.info('fetch status of ' + quad_uri)
            try:
                status = QUAD_URI_TO_STATUS_MAP[quad_uri]
                if status == 'error':
                    raise Exception('error on %s', quad_uri)
                status_list.append(status)
                if isinstance(status, list):
                    # success, delete from record
                    del QUAD_URI_TO_STATUS_MAP[quad_uri]
            except KeyError:
                status_list.append('not found')
        return {'status_list': status_list}
    except Exception:
        LOGGER.exception('something went wrong')
        global HEALTHY
        HEALTHY = False
        raise


if __name__ == '__main__':
    """Entry point."""
    parser = argparse.ArgumentParser(description='Tensorflow inference server')
    parser.add_argument('tensorflow_model_path', help='path to frozen model')
    parser.add_argument(
        '--app_port', default=80, help='server port')
    parser.add_argument('--gpu', help='which GPU to use')
    args = parser.parse_args()

    check_keras_version()
    check_tf_version()

    # optionally choose specific GPU
    if args.gpu:
        setup_gpu(args.gpu)

    LOGGER.info(f'loading {args.tensorflow_model_path}')
    model = models.load_model(
        args.tensorflow_model_path, backbone_name='resnet50')

    quad_offset_queue = queue.Queue()
    quad_file_path_queue = queue.Queue()

    # make clipper workers
    for _ in range(multiprocessing.cpu_count()):
        quad_processor_worker_thread = threading.Thread(
            target=quad_processor,
            args=(quad_offset_queue, quad_file_path_queue))
        quad_processor_worker_thread.daemon = True
        quad_processor_worker_thread.start()

    inference_lock = threading.Lock()
    for _ in range(JOBS_PER_WORKER):
        do_inference_worker_thread = threading.Thread(
            target=do_inference_worker,
            args=(model, quad_offset_queue, quad_file_path_queue,
                  inference_lock))
        do_inference_worker_thread.daemon = True
        do_inference_worker_thread.start()
    APP.run(host='0.0.0.0', port=args.app_port)
