import argparse
import json
import os
import keras
import keras.backend as K

import tensorflow as tf
from keras.engine.saving import load_model
from keras_applications.mobilenet_v2 import relu6
from tensorflow.python.framework import graph_util, graph_io

from utils.utils import add_regression_layer_if_not_exists
from yolo import RegressionLayer

model_name = '{}_model.h5'


def is_tiny_model(model_name):
    return model_name in ['tiny_yolo3', 'mobilenet2']


def _main_(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    K.set_learning_phase(0)

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    snapshot_name = os.path.join(config['inference']['snapshots_path'],
                                 model_name.format(config['model']['type']))
    is_tiny = is_tiny_model(config['model']['type'])
    anchors = config['model']['anchors'] if not is_tiny else config['model']['tiny_anchors']

    custom_objects = {'RegressionLayer': RegressionLayer}
    if config['inference']['is_mobilenet2']:
        custom_objects = {'relu6': relu6}
    model = load_model(snapshot_name, custom_objects=custom_objects)
    model = add_regression_layer_if_not_exists(model, anchors)
    model.summary()

    if type(model.output) is list:
        output_names = [o.name.split(':')[0] for o in model.output]
    else:
        output_names = [model.output.name.split(':')[0]]
    print('TF version: {}'.format(tf.__version__))
    print('Save graph with outputs: {}'.format(output_names))

    sess = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_names)
    graph_io.write_graph(constant_graph,
                         ".",
                         snapshot_name + '.pb',
                         as_text=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Freeze keras model to tf graph')

    argparser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')

    argparser.add_argument(
        '-o',
        '--output',
        help='output path for frozen graph')

    args = argparser.parse_args()
    _main_(args)
