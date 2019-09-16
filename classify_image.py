"""Script to classify a single image."""
import sys
import logging

import PIL
import tensorflow as tf
import numpy



MODEL_PATH = 'models/fasterRCNN_08-26-withnotadams.pb'
IMAGE_PATH = 'test_data/11133.png'
LOGGING_LEVEL = logging.DEBUG

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
    image_array = numpy.array(image)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_array_expanded = numpy.expand_dims(image_array, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            box = detection_graph.get_tensor_by_name('detection_boxes:0')
            score = detection_graph.get_tensor_by_name('detection_scores:0')
            clss = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            LOGGER.debug((image_array_expanded.shape, image_tensor, box, score, clss, num_detections))
            # Actual detection
            (box, score, clss, num_detections) = sess.run(
                    [box, score, clss, num_detections],
                    feed_dict={image_tensor: image_array_expanded})
            LOGGER.debug((box, score, clss, num_detections))


if __name__ == '__main__':
    DETECTION_GRAPH = load_model(MODEL_PATH)
    LOGGER.debug(DETECTION_GRAPH)
    do_detection(DETECTION_GRAPH, IMAGE_PATH)
