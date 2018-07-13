import argparse
import json
import os
import keras
import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io


model_name = '{}_model.h5'


def _main_(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    K.set_learning_phase(0)

    config_path = args.conf
    output = args.output
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    snapshot_name = os.path.join(config["inference"]['snapshots_path'],
                                 model_name.format(config['model']['type']))

    model = keras.models.load_model(snapshot_name)
    model.summary()

    output_names = [o.name.split(':')[0] for o in model.output]
    print('TF version: {}'.format(tf.__version__))
    print('Save graph with outputs: {}'.format(output_names))

    sess = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_names)
    graph_io.write_graph(constant_graph,
                         ".",
                         os.path.join(output, snapshot_name + '.pb'),
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
