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

    train_datasets = [{**ds, 'path': os.path.join(config['train']['images_dir'], ds['path'])}
                      for ds in config['train']['train_datasets']]
    validation_datasets = [{**ds, 'path': os.path.join(config['train']['images_dir'], ds['path'])}
                           for ds in config['train']['validation_datasets']]

    trassir_annotation = TrassirRectShapesAnnotations(train_datasets, validation_datasets)
    trassir_annotation.load()
    trassir_annotation.print_statistics()
    train = trassir_annotation.get_train_instances(config['model']['labels'],
                                                   config['train']['verifiers'],
                                                   config['model']['max_box_per_image'])
    print('Train len: ', len(train))
    validation = trassir_annotation.get_validation_instances(config['model']['labels'],
                                                             config['train']['verifiers'],
                                                             config['model']['max_box_per_image'])
    print('Val len: ', len(validation))

    generator = BatchGenerator(
        instances=train,
        anchors=config['model']['anchors'],
        labels=config['model']['labels'],
        downsample=32,
        max_box_per_image=config['model']['max_box_per_image'],
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=jitter,
        norm=None
    )

    for i in range(len(generator)):
        for image in generator[i][0][0]:
            cv2.imshow('image', cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))
            key = cv2.waitKeyEx(0)
            if key == 27:
                return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Batch generator test')

    argparser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
    argparser.add_argument(
        '-j',
        '--jitter',
        help='augmentation strength')

    args = argparser.parse_args()
    _main_(args)
