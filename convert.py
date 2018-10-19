#! /usr/bin/env python

import os
import keras

from utils.utils import add_regression_layer_if_not_exists
from keras.models import load_model

from yolo import RegressionLayer

model_name = '{}_model.h5'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


keras.backend.set_learning_phase(0)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

anchors = [10,37, 17,71, 28,104, 28,50, 42,79, 45,148, 70,92, 77,181, 193,310]
input_path = 'snapshots/current_head/yolo3_model.h5'
output_path = 'snapshots/current_head/yolo3_model.h5'

custom_objects = {'RegressionLayer': RegressionLayer}
infer_model = load_model(input_path, custom_objects=custom_objects)
infer_model = add_regression_layer_if_not_exists(infer_model, anchors)

keras.models.save_model(infer_model, output_path)
