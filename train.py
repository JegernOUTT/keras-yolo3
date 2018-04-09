#! /usr/bin/env python

import argparse
import json
import os
import shutil

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam

from callback import EvaluateCallback
from generator import BatchGenerator
from preprocessing import load_images
from utils.utils import normalize, evaluate
from yolo import create_yolov3_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and evaluate YOLO_v3 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def create_callbacks(config, infer_model):
    monitor_metric, metric_mode = 'val_loss', 'min'
    if 'yolo_3' in os.listdir(os.path.expanduser('~/logs/')):
        shutil.rmtree(os.path.join(os.path.expanduser('~/logs/'), 'yolo_3'))

    evaluate_callback = EvaluateCallback(
        infer_model=infer_model,
        anchors=config['model']['anchors'],
        labels=config['model']['labels']
    )

    early_stop = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.001,
        patience=20,
        mode=metric_mode,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        os.path.join(config['train']['saved_weights_name'], 'best_weights.h5'),
        monitor=monitor_metric,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode=metric_mode,
        period=1)

    tensorboard = TensorBoard(
        log_dir=os.path.expanduser('~/logs/yolo_3'),
        batch_size=config['train']['batch_size'],
        write_graph=True,
        write_images=True)

    reduce_lrt = ReduceLROnPlateau(
        monitor=monitor_metric,
        verbose=1,
        patience=5,
        mode=metric_mode,
        min_lr=1e-07,
        factor=0.8)

    return [early_stop, checkpoint, tensorboard, reduce_lrt, evaluate_callback]


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
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

    train_model, infer_model = create_yolov3_model(
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=config['model']['max_box_per_image'],
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh']
    )

    # load the weight of the backend, which includes all layers but the last ones
    if os.path.exists(config['train']['pretrained_weights']) and os.path.isfile(config['train']['pretrained_weights']):
        print("Loading pretrained weights: {}".format(config['train']['pretrained_weights']))
        train_model.load_weights(config['train']['pretrained_weights'], by_name=True)
    else:
        train_model.load_weights("backend.h5", by_name=True)

    ###############################
    #   Kick off the training
    ###############################
    optimizer = Adam(lr=config['train']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer)

    callbacks = create_callbacks(config, infer_model)

    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=1 if config['train']['debug'] else 1,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=callbacks,
        workers=8,
        max_queue_size=16
    )

    infer_model.load_weights(os.path.join(config['train']['saved_weights_name'], 'best_weights.h5'), by_name=True)
    infer_model.save(os.path.join(config['train']['saved_weights_name'], 'yolo3.h5'))

    ###############################
    #   Run the evaluation
    ###############################
    infer_model = load_model(os.path.join(config['train']['saved_weights_name'], 'yolo3.h5'))

    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
