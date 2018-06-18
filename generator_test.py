import os
import logging
import json

import numpy as np
import matplotlib.pyplot as plt

from generator import BatchGenerator
from preprocessing import TrassirAnnotations


def imshow_grid(data, height=None, width=None, normalize=False, padsize=1, padval=0):
    if normalize:
        data -= data.min()
        data /= data.max()

    N = data.shape[0]
    if height is None:
        if width is None:
            height = int(np.ceil(np.sqrt(N)))
        else:
            height = int(np.ceil(N / float(width)))

    if width is None:
        width = int(np.ceil(N / float(height)))

    assert height * width >= N

    # append padding
    padding = ((0, (width * height) - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((height, width) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((height * data.shape[1], width * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.show()


logging.basicConfig(level=logging.INFO)

config_path = 'config.json'
assert os.path.isfile(config_path), "Check config path: {}".format(config_path)
with open(config_path, 'r') as f:
    config = json.load(f)

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
    jitter=0.0,
    norm=None
)

plt.rcParams['figure.figsize'] = (15, 15)

for i in range(10):
    for image in generator[i][0][0]:
        imshow_grid(np.expand_dims(image, axis=0), height=1, width=1, normalize=True)
