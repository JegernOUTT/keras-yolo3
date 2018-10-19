#! /usr/bin/env python

import argparse
import json
import os

from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from tqdm import tqdm

from generator import BatchGenerator
from preprocessing import TrassirRectShapesAnnotations
from utils.utils import normalize
from yolo import create_full_model, dummy_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
logs_dir = '~/logs/'
model_filename = '{}_model.h5'
weights_filename = '{}_weights.h5'


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

    losses = []
    input_size = 416

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    is_tiny = is_tiny_model(config['model']['type'])
    anchors = config['model']['anchors'] if not is_tiny else config['model']['tiny_anchors']

    train_datasets = [{**ds, 'path': os.path.join(config['train']['images_dir'], ds['path'])}
                      for ds in config['train']['train_datasets']]

    trassir_annotation = TrassirRectShapesAnnotations(train_datasets, [], config['model']['labels'], config['model']['skip_labels'])
    trassir_annotation.load()
    trassir_annotation.print_statistics()
    train = trassir_annotation.get_train_instances(config['train']['verifiers'],
                                                   config['model']['max_box_per_image'])

    print('There is {} training instances'.format(len(train)))

    if not os.path.exists(config['train']['snapshots_path']):
        os.mkdir(config['train']['snapshots_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        model_type=config['model']['type'],
        freeze_base_model=config['model']['need_to_freeze_base'],
        nb_class=len(config['model']['labels']),
        anchors=anchors,
        max_box_per_image=config['model']['max_box_per_image'],
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=1,
        warmup_batches=0,
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

    for image in tqdm(train):
        generator = BatchGenerator(
            instances=[image],
            anchors=anchors,
            labels=config['model']['labels'],
            downsample=32,
            max_box_per_image=config['model']['max_box_per_image'],
            batch_size=1,
            min_net_size=input_size,
            max_net_size=input_size,
            shuffle=True,
            jitter=0.0,
            norm=normalize,
            advanced_aug=False
        )
        losses.append(
            {
                'image': image['file_name'],
                'losses': train_model.evaluate_generator(generator=generator)
            }
        )
    with open('losses.json', 'w') as f:
        losses = sorted(losses, reverse=True, key=lambda x: x['losses'][0])
        json.dump(losses, f, indent=2)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Train and evaluate YOLO_v3 model on any dataset')

    argparser.add_argument(
        '-c',
        '--conf',
        default='config.json',
        help='path to configuration file')

    argparser.add_argument('-hvd',
        action='store_true',
        default=False,
        dest='horovod',
        help='use this option for mpirun')

    args = argparser.parse_args()
    _main_(args)
