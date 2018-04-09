#! /usr/bin/env python

import argparse
import os
import numpy as np
import json

from preprocessing import load_images
from voc import parse_voc_annotation
from yolo import create_yolov3_model
from generator import BatchGenerator
from utils.utils import normalize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and evaluate YOLO_v3 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def create_callbacks(saved_weights_name):
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        mode='min',
        verbose=1
    )

    checkpoint = ModelCheckpoint(os.path.join(saved_weights_name, 'weights.{epoch:02d}-{val_loss:.2f}.h5'),
                                 monitor='val_loss', verbose=1, save_best_only=False,
                                 save_weights_only=True, mode='min', period=2)

    return [early_stop, checkpoint]


def filter_categories(data_train, data_val, categories_to_filter):
    # TODO: fix this
    categories_to_filter = dict(zip(range(len(categories_to_filter)), categories_to_filter))

    data_train['images_with_annotations'] = [
        (image, [a for a in ann if a['category_id'] in categories_to_filter])
        for image, ann in data_train['images_with_annotations']
    ]

    data_val['images_with_annotations'] = [
        (image, [a for a in ann if a['category_id'] in categories_to_filter])
        for image, ann in data_val['images_with_annotations']
    ]


def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    labels = config['model']['labels']
    train_ints, valid_ints = load_images(config)
    filter_categories(train_ints, valid_ints, labels)

    ###############################
    #   Create the generators 
    ###############################    
    train_generator = BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        labels=config['model']['labels'],
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=config['model']['max_box_per_image'],
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.3,
        norm=normalize
    )

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=config['model']['labels'],
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=config['model']['max_box_per_image'],
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )

    ###############################
    #   Create the model 
    ###############################
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator) + \
                                                         config['valid']['valid_times'] * len(valid_generator))

    train_model, infer_model = create_yolov3_model(
        nb_class=len(config['model']['labels']),
        anchors=config['model']['anchors'],
        max_box_per_image=config['model']['max_box_per_image'],
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh']
    )

    train_model.load_weights("backend.h5", by_name=True)

    ###############################
    #   Kick off the training
    ###############################
    optimizer = Adam(lr=config['train']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer)

    callbacks = create_callbacks(config['train']['saved_weights_name'])

    # train_model.fit_generator(
    #     generator=train_generator,
    #     steps_per_epoch=len(train_generator) * config['train']['train_times'],
    #     epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
    #     verbose=1 if config['train']['debug'] else 1,
    #     validation_data=valid_generator,
    #     validation_steps=len(valid_generator) * config['valid']['valid_times'],
    #     callbacks=callbacks,
    #     workers=8,
    #     max_queue_size=16
    # )

    infer_model.load_weights(config['train']['saved_weights_name'], by_name=True)
    infer_model.save('yolo3.h5')


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
