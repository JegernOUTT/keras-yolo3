import argparse

import cv2
import os
import logging
import json

import numpy as np

from generator import BatchGenerator
from preprocessing import TrassirRectShapesAnnotations


logging.basicConfig(level=logging.INFO)


def _main_(args):
    config_path = args.conf
    jitter = float(args.jitter)

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    validation_datasets = [{**ds, 'path': os.path.join(config['train']['images_dir'], ds['path'])}
                           for ds in config['train']['train_datasets']]

    trassir_annotation = TrassirRectShapesAnnotations([], validation_datasets, config['model']['labels'], config['model']['skip_labels'])
    trassir_annotation.load()
    trassir_annotation.print_statistics()
    validation = trassir_annotation.get_validation_instances(config['train']['verifiers'],
                                                             config['model']['max_box_per_image'])
    print('Val len: ', len(validation))

    generator = BatchGenerator(
        instances=validation,
        anchors=config['model']['anchors'],
        labels=config['model']['labels'],
        downsample=32,
        max_box_per_image=config['model']['max_box_per_image'],
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=jitter,
        norm=None,
        advanced_aug=True
    )

    for i in range(len(generator)):
        for image in generator[i][0][0]:
            cv2.imshow('image', cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))
            key = cv2.waitKeyEx(0) & 0xFF
            if key == 27:
                return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Batch generator test')

    argparser.add_argument(
        '-c',
        '--conf',
        default='config.json',
        help='path to configuration file')
    argparser.add_argument(
        '-j',
        '--jitter',
        default=0.0,
        help='augmentation strength')

    args = argparser.parse_args()
    _main_(args)
