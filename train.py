#! /usr/bin/env python

import argparse
import json
import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

from callback import CustomModelCheckpoint, CustomTensorBoard
from generator import BatchGenerator
from preprocessing import TrassirAnnotations
from utils.multi_gpu_model import multi_gpu_model
from utils.utils import normalize, evaluate
from yolo import create_full_model, dummy_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
logs_dir = '~/logs/'
model_filename = '{}_model.h5'
weights_filename = '{}_weights.h5'


def create_callbacks(config, infer_model, validation_generator):
    global logs_dir, model_filename, weights_filename

    logdir_prefix = config['train']['log_prefix']
    model_type = config['model']['type']
    monitor_metric, metric_mode = 'val_loss', 'min'
    model_log_path = os.path.join(model_type, '_'.join(config['model']['labels']),
                                  '_{}'.format(logdir_prefix) if logdir_prefix != '' else '')
    if model_log_path in os.listdir(os.path.expanduser(logs_dir)):
        shutil.rmtree(os.path.join(os.path.expanduser(logs_dir), model_name))

    checkpoint_weights = CustomModelCheckpoint(
        model_to_save=infer_model,
        filepath=os.path.join(config['train']['snapshots_path'], weights_filename.format(model_type)),
        monitor=monitor_metric,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode=metric_mode,
        period=1)

    checkpoint_model = CustomModelCheckpoint(
        model_to_save=infer_model,
        filepath=os.path.join(config['train']['snapshots_path'], model_filename.format(model_type)),
        monitor=monitor_metric,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode=metric_mode,
        period=1)

    tensorboard_map = CustomTensorBoard(
        infer_model=infer_model,
        validation_generator=validation_generator,
        log_dir=os.path.expanduser(os.path.join(logs_dir, model_log_path)),
        write_graph=True,
        write_images=True)

    reduce_lrt = LearningRateScheduler(
        lambda x: config['train']['learning_rate'] if x < 30 else config['train']['learning_rate'] * 0.1)

    return [
        checkpoint_weights,
        checkpoint_model,
        tensorboard_map,
        reduce_lrt,
    ]


def create_model(
        model_type,
        freeze_base_model,
        nb_class,
        anchors,
        max_box_per_image,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        multi_gpu,
        saved_weights_name,
        yolo_loss_options,
        model_scale_coefficient,
        debug_loss
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            train_model, infer_model = create_full_model(
                model_type=model_type,
                freeze_base_model=freeze_base_model,
                nb_class=nb_class,
                anchors=anchors,
                max_box_per_image=max_box_per_image,
                max_grid=max_grid,
                batch_size=batch_size // multi_gpu,
                warmup_batches=warmup_batches,
                ignore_thresh=ignore_thresh,
                yolo_loss_options=yolo_loss_options,
                model_scale_coefficient=model_scale_coefficient,
                debug_loss=debug_loss
            )
    else:
        train_model, infer_model = create_full_model(
            model_type=model_type,
            freeze_base_model=freeze_base_model,
            nb_class=nb_class,
            anchors=anchors,
            max_box_per_image=max_box_per_image,
            max_grid=max_grid,
            batch_size=batch_size // multi_gpu,
            warmup_batches=warmup_batches,
            ignore_thresh=ignore_thresh,
            yolo_loss_options=yolo_loss_options,
            model_scale_coefficient=model_scale_coefficient,
            debug_loss=debug_loss
        )

    train_model.summary()

    if os.path.exists(saved_weights_name):
        print('\nLoading from pretrained weights: {}\n'.format(saved_weights_name))
        train_model.load_weights(saved_weights_name)
    else:
        if os.path.exists('./scratch_model.h5'):
            print('\nLoading from scratch model: {}\n'.format('./scratch_model.h5'))
            train_model.load_weights('./scratch_model.h5', by_name=True)
        else:
            print('\nLoading blank model\n')

    if multi_gpu > 1:
        train_model = multi_gpu_model(train_model, gpus=multi_gpu)

    return train_model, infer_model


def is_tiny_model(model_name):
    return model_name in ['tiny_yolo3', 'mobilenet2']


def _main_(args):
    config_path = args.conf
    global weights_name, model_name

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    is_tiny = is_tiny_model(config['model']['type'])
    anchors = config['model']['anchors'] if not is_tiny else config['model']['tiny_anchors']

    train_datasets = [{**ds, 'path': os.path.join(config['train']['images_dir'], ds['path'])}
                      for ds in config['train']['train_datasets']]
    validation_datasets = [{**ds, 'path': os.path.join(config['train']['images_dir'], ds['path'])}
                           for ds in config['train']['validation_datasets']]

    trassir_annotation = TrassirAnnotations(train_datasets, validation_datasets)
    trassir_annotation.load()
    trassir_annotation.print_statistics()
    train = trassir_annotation.get_train_instances(config['model']['labels'],
                                                   config['train']['verifiers'],
                                                   config['model']['max_box_per_image'])
    validation = trassir_annotation.get_validation_instances(config['model']['labels'],
                                                             config['train']['verifiers'],
                                                             config['model']['max_box_per_image'])

    print('There is {} training instances'.format(len(train)))
    print('There is {} validation instances'.format(len(validation)))

    if not os.path.exists(config['train']['snapshots_path']):
        os.mkdir(config['train']['snapshots_path'])

    train_generator = BatchGenerator(
        instances=train,
        anchors=anchors,
        labels=config['model']['labels'],
        downsample=32,
        max_box_per_image=config['model']['max_box_per_image'],
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.1,
        norm=normalize
    )

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

    if os.path.exists(os.path.join(config['train']['snapshots_path'],
                                   weights_filename.format(config['model']['type']))):
        warmup_batches = 0
    else:
        warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        model_type=config['model']['type'],
        freeze_base_model=config['model']['need_to_freeze_base'],
        nb_class=len(config['model']['labels']),
        anchors=anchors,
        max_box_per_image=config['model']['max_box_per_image'],
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh'],
        multi_gpu=multi_gpu,
        saved_weights_name=os.path.join(config['train']['snapshots_path'],
                                        weights_filename.format(config['model']['type'])),
        yolo_loss_options=config["loss_config"]["yolo_loss"],
        model_scale_coefficient=config["model"]["model_scale_coefficient"],
        debug_loss=config["loss_config"]["debug_loss"]
    )

    optimizer = Adam(
        lr=config['train']['learning_rate']
    )

    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    callbacks = create_callbacks(config, infer_model, valid_generator)

    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=1,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=callbacks,
        workers=4,
        max_queue_size=16
    )

    infer_model.load_weights(os.path.join(config['train']['snapshots_path'],
                                          weights_filename.format(config['model']['type'])))
    recalls, average_precisions = evaluate(infer_model, valid_generator)

    for (label, average_precision), (_, recall) in zip(average_precisions.items(), recalls.items()):
        print('{}: mAP {:.4f}, recall: {:.4f}'.format(config['model']['labels'][label], average_precision, recall))
    print('mAP: {:.4f}, recall: {:.4f}'.format(np.mean(list(average_precisions.values())),
                                               np.mean(list(recalls.values()))))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Train and evaluate YOLO_v3 model on any dataset')

    argparser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
