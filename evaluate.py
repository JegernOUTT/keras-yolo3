#! /usr/bin/env python

import argparse
import json
import os

from keras.models import load_model
from keras_applications.mobilenet_v2 import relu6

from generator import BatchGenerator
from preprocessing import TrassirRectShapesAnnotations
from utils.utils import normalize, evaluate, add_regression_layer_if_not_exists
from yolo import RegressionLayer

model_name = '{}_model.h5'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def is_tiny_model(model_name):
    return model_name in ['tiny_yolo3', 'mobilenet2']


def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    os.environ['CUDA_VISIBLE_DEVICES'] = config['inference']['gpu']
    is_tiny = is_tiny_model(config['model']['type'])
    snapshot_name = os.path.join(config['inference']['snapshots_path'], model_name.format(config['model']['type']))
    anchors = config['model']['anchors'] if not is_tiny else config['model']['tiny_anchors']

    validation_datasets = [{**ds, 'path': os.path.join(config['train']['images_dir'], ds['path'])}
                           for ds in config['train']['validation_datasets']]

    trassir_annotation = TrassirRectShapesAnnotations([], validation_datasets, config['model']['labels'], config['model']['skip_labels'])
    trassir_annotation.load()
    trassir_annotation.print_statistics()
    validation = trassir_annotation.get_validation_instances(config['train']['verifiers'],
                                                             config['model']['max_box_per_image'])

    print('There is {} validation instances'.format(len(validation)))

    valid_generator = BatchGenerator(
        instances=validation,
        anchors=anchors,
        labels=config['model']['labels'],
        downsample=32,
        max_box_per_image=config['model']['max_box_per_image'],
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )

    custom_objects = {'RegressionLayer': RegressionLayer}
    if config['inference']['is_mobilenet2']:
        custom_objects = {'relu6': relu6}
    infer_model = load_model(snapshot_name, custom_objects=custom_objects)
    infer_model = add_regression_layer_if_not_exists(infer_model, anchors)

    recalls, average_precisions = evaluate(infer_model, valid_generator)

    for label, average_precision in average_precisions.items():
        print(config['model']['labels'][label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument(
    	'-c',
    	'--conf',
    	default='config.json',
    	help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
