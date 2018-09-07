from typing import Tuple, Union

import keras
import keras_resnet.models
import keras.backend as K
import keras.layers as KL
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Input
from keras.models import Model

IntOrNone = Union[int, None]


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def create_classifier_model(image_shape: Tuple[IntOrNone, IntOrNone, int],
                            classes_count: int) -> keras.Model:
    return keras.applications.ResNet50(input_shape=image_shape, weights=None, classes=classes_count)
