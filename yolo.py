from keras.applications import MobileNetV2
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Lambda, \
    regularizers, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
from keras.regularizers import l2
from functools import reduce
import operator
import numpy as np


class RegressionLayer(Layer):
    def __init__(self, anchors=[], **kwargs):
        self.anchors_list = anchors if type(anchors) is list else anchors.tolist()
        self.anchors = K.variable(anchors, dtype=K.floatx(), name='anchors')
        super(RegressionLayer, self).__init__(**kwargs)
        
    def set_anchors(self, anchors):
        self.anchors = K.variable(anchors, dtype=K.floatx(), name='anchors')

    def build(self, input_shape):
        super(RegressionLayer, self).build(input_shape)

    def call(self, x):
        input, output_layers = x[0], x[1:]

        net_h, net_w = K.shape(input)[1], K.shape(input)[2]
        net_factor = K.reshape(K.cast([net_w, net_h], dtype=K.floatx()), [1, 1, 1, 1, 2])

        batch_size = K.shape(output_layers[0])[0]
        last_dim = K.shape(output_layers[0])[3] // 3

        result = None
        for i in range(len(output_layers)):
            max_net_h, max_net_w = net_h * (2 ** i), net_w * (2 ** i)

            cell_x = K.cast(K.reshape(
                K.tile(tf.range(max_net_w), [max_net_h]), (1, max_net_h, max_net_w, 1, 1)), dtype=K.floatx())
            cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
            cell_grid = K.cast(K.tile(K.concatenate([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1]),
                               dtype=K.floatx())

            current_anchors = self.anchors[(len(output_layers) - i - 1) * 6: (len(output_layers) - i) * 6]
            current_anchors = K.reshape(current_anchors, [1, 1, 1, 3, 2])

            current_pred = K.reshape(output_layers[i],
                                     tf.concat([tf.shape(output_layers[i])[:3], tf.constant([3, -1])], axis=0))

            pred_box_conf = K.expand_dims(tf.sigmoid(current_pred[..., 4]), 4)
            pred_box_class = current_pred[..., 5:]

            grid_h, grid_w = K.shape(current_pred)[1], K.shape(current_pred)[2]
            grid_factor = K.reshape(K.cast([grid_w, grid_h], dtype=K.floatx()), [1, 1, 1, 1, 2])

            pred_box_xy = (cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(current_pred[..., :2]))
            pred_xy = K.expand_dims(pred_box_xy / grid_factor, 4)
            pred_xy = K.reshape(pred_xy, [batch_size, grid_h, grid_w, 3, 2])

            pred_box_wh = current_pred[..., 2:4]
            pred_wh = K.expand_dims(tf.exp(pred_box_wh) * current_anchors / net_factor, 4)
            pred_wh = K.reshape(pred_wh, [batch_size, grid_h, grid_w, 3, 2])

            if result is None:
                current_result = K.concatenate([pred_xy, pred_wh, pred_box_conf, pred_box_class], axis=4)
                current_result_shape = K.shape(current_result)
                current_result = K.reshape(current_result,
                                           [batch_size,
                                            grid_h * grid_w * current_result_shape[3],
                                            last_dim])
                result = current_result

            else:
                current_result = K.concatenate([pred_xy, pred_wh, pred_box_conf, pred_box_class], axis=4)
                current_result_shape = K.shape(current_result)
                current_result = K.reshape(current_result,
                                           [batch_size,
                                            grid_h * grid_w * current_result_shape[3],
                                            last_dim])
                result = K.concatenate([result, current_result], axis=1)

        result = K.identity(result, name='output')
        return result

    def compute_output_shape(self, input_shape):
        if input_shape[0][1] is not None:
            second_dim = reduce(operator.add, [reduce(operator.mul, [*i[1:3], 3], 1) for i in input_shape])
        else:
            second_dim = None

        return input_shape[0][0], second_dim, input_shape[0][3] // 3

    def get_config(self):
        config = super().get_config()
        config.update({'anchors': self.anchors_list})
        return config


class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh,
                 yolo_loss_options, debug_loss, **kwargs):
        # make the model settings persistent
        self.ignore_thresh = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors = tf.constant(anchors, dtype=K.floatx(), shape=[1, 1, 1, 3, 2])

        self.grid_scale = yolo_loss_options['grid_scale']
        self.obj_scale = yolo_loss_options['obj_scale']
        self.noobj_scale = yolo_loss_options['noobj_scale']
        self.xywh_scale = yolo_loss_options['xywh_scale']
        self.class_scale = yolo_loss_options['class_scale']
        self.debug_loss = debug_loss

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)

        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], K.floatx()), [1, 1, 1, 1, 2])

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(tf.cast([net_w, net_h], K.floatx()), [1, 1, 1, 1, 2])

        """
        Adjust prediction
        """
        pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh = y_pred[..., 2:4]  # t_wh
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)  # adjust confidence
        pred_box_class = y_pred[..., 5:]  # adjust class probabilities

        """
        Adjust ground truth
        """
        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = y_true[..., 2:4]  # t_wh
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Compare each predicted box to all true boxes
        """
        # initially, drag all objectness of all boxes to 0
        conf_delta = pred_box_conf - 0

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)
        iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

        count = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf * object_mask) >= 0.5)
        class_mask = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50 = tf.reduce_sum(tf.to_float(iou_scores >= 0.5) * detect_mask * class_mask) / (count + 1e-3)
        recall75 = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask * class_mask) / (count + 1e-3)
        avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj = tf.reduce_sum(pred_box_conf * object_mask) / (count + 1e-3)
        avg_noobj = tf.reduce_sum(pred_box_conf * (1 - object_mask)) / (count_noobj + 1e-3)
        avg_cat = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)

        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches + 1),
                                                      lambda: [true_box_xy + (
                                                              0.5 + self.cell_grid[:, :grid_h, :grid_w, :, :]) * (
                                                                       1 - object_mask),
                                                               true_box_wh + tf.zeros_like(true_box_wh) * (
                                                                       1 - object_mask),
                                                               tf.ones_like(object_mask)],
                                                      lambda: [true_box_xy,
                                                               true_box_wh,
                                                               object_mask])

        """
        Compare each true box to all anchor boxes
        """
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)  # the smaller the box, the bigger the scale

        xy_delta = xywh_mask * (pred_box_xy - true_box_xy) * wh_scale * self.xywh_scale
        wh_delta = xywh_mask * (pred_box_wh - true_box_wh) * wh_scale * self.xywh_scale
        conf_delta = object_mask * (pred_box_conf - true_box_conf) * self.obj_scale + (
                    1 - object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(
                          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class),
                          4) * \
                      self.class_scale

        loss_xy = tf.reduce_mean(tf.square(xy_delta), list(range(1,5)))
        loss_wh = tf.reduce_mean(tf.square(wh_delta), list(range(1,5)))
        loss_conf = tf.reduce_mean(tf.square(conf_delta), list(range(1,5)))
        loss_class = tf.reduce_mean(class_delta, list(range(1,5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        if self.debug_loss:
            loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
            loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
            loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
            loss = tf.Print(loss, [grid_h, loss], message='loss xy, wh, conf, class: \t', summarize=1000)

        return loss * self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def __bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    ''' Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = input

    grouped_channels = int(filters / cardinality)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='valid' if strides > 1 else 'same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:
        if init._keras_shape[-1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='valid' if strides > 1 else 'same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding='valid' if strides > 1 else 'same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding='valid' if strides > 1 else 'same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    x = add([init, x])
    x = LeakyReLU(alpha=0.1)(x)

    return x


def _conv_block(inp, coef, convs, resnext=False, do_skip=True):
    x = inp
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and do_skip:
            skip_connection = x
        count += 1

        if conv['stride'] > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)

        if resnext:
            # Now work only with strides == 1 #
            assert conv['stride'] == 1
            x = __bottleneck_block(input=x,
                                   filters=int(conv['filter'] * coef) if not 'no_scale' in conv else conv['filter'],
                                   strides=conv['stride'])
            return add([skip_connection, x]) if do_skip else x
        else:
            x = Conv2D(int(conv['filter'] * coef) if not 'no_scale' in conv else conv['filter'],
                       conv['kernel'],
                       strides=conv['stride'],
                       padding='valid' if conv['stride'] > 1 else 'same',
                       name='conv_' + str(conv['layer_idx']),
                       use_bias=False if conv['bnorm'] else True,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform')(x)
            if conv['bnorm']:
                x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
            if conv['leaky']:
                x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return add([skip_connection, x]) if do_skip else x


def _make_yolo3_model(
        input_shape,
        max_box_per_image,
        nb_class,
        anchors,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        yolo_loss_options,
        debug_loss,
        model_scale_coefficient=1):
    input = Input(shape=input_shape, name='input_1')
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_2')
    # grid_h, grid_w, number of box in one anchor, 5+nb_class
    true_yolo_1 = Input(shape=(None, None, 3, 4 + 1 + nb_class), name='input_3')
    true_yolo_2 = Input(shape=(None, None, 3, 4 + 1 + nb_class), name='input_4')
    true_yolo_3 = Input(shape=(None, None, 3, 4 + 1 + nb_class), name='input_5')

    x = _conv_block(input, model_scale_coefficient,
                    [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                     {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                     {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                     {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    x = _conv_block(x, model_scale_coefficient,
                    [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                     {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                     {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    x = _conv_block(x, model_scale_coefficient,
                    [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                     {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    x = _conv_block(x, model_scale_coefficient,
                    [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                     {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                     {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    for i in range(7):
        x = _conv_block(x, model_scale_coefficient,
                        [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                          'layer_idx': 16 + i * 3},
                         {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                          'layer_idx': 17 + i * 3}])

    skip_36 = x
    x = _conv_block(x, model_scale_coefficient,
                    [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                     {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                     {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    for i in range(7):
        x = _conv_block(x, model_scale_coefficient,
                        [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                          'layer_idx': 41 + i * 3},
                         {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                          'layer_idx': 42 + i * 3}])

    skip_61 = x
    x = _conv_block(x, model_scale_coefficient,
                    [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                     {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                     {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    for i in range(3):
        x = _conv_block(x, model_scale_coefficient,
                        [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                          'layer_idx': 66 + i * 3},
                         {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                          'layer_idx': 67 + i * 3}])

    last = _conv_block(x, model_scale_coefficient,
                       [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}],
                       do_skip=False)

    # Feature Pyramid Network with 2 heads
    # First branch
    pred_yolo_1 = _conv_block(last, model_scale_coefficient,
                              [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80},
                               {'filter': (3 * (5 + nb_class)), 'no_scale': None, 'kernel': 1, 'stride': 1,
                                'bnorm': False, 'leaky': False, 'layer_idx': 81}],
                              do_skip=False)
    loss_yolo_1 = YoloLayer(anchors[12:],
                            [1 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            {**yolo_loss_options, 'grid_scale': yolo_loss_options['grid_scales'][0]},
                            debug_loss) \
        ([input, pred_yolo_1, true_yolo_1, true_boxes])

    # Second branch
    x = _conv_block(last, 1,
                    [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}],
                    do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    x = _conv_block(x, 1,
                    [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                     {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                     {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                     {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                     {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}],
                    do_skip=False)

    pred_yolo_2 = _conv_block(x, 1,
                              [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 92},
                               {'filter': (3 * (5 + nb_class)), 'no_scale': None, 'kernel': 1, 'stride': 1,
                                'bnorm': False, 'leaky': False, 'layer_idx': 93}],
                              do_skip=False)
    loss_yolo_2 = YoloLayer(anchors[6:12],
                            [2 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            {**yolo_loss_options, 'grid_scale': yolo_loss_options['grid_scales'][1]},
                            debug_loss) \
        ([input, pred_yolo_2, true_yolo_2, true_boxes])

    # Third branch
    x = _conv_block(x, 1,
                    [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96}],
                    do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    pred_yolo_3 = _conv_block(x, 1,
                              [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 104},
                               {'filter': (3 * (5 + nb_class)), 'no_scale': None, 'kernel': 1, 'stride': 1,
                                'bnorm': False, 'leaky': False, 'layer_idx': 105}],
                              do_skip=False)

    loss_yolo_3 = YoloLayer(anchors[:6],
                            [4 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            {**yolo_loss_options, 'grid_scale': yolo_loss_options['grid_scales'][2]},
                            debug_loss) \
        ([input, pred_yolo_3, true_yolo_3, true_boxes])

    return {
        'base_model': Model(input, last),
        'losses': [loss_yolo_1, loss_yolo_2, loss_yolo_3],
        'train_model': Model([input, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3],
                             [loss_yolo_1, loss_yolo_2, loss_yolo_3]),
        'infer_model': Model(input, RegressionLayer(anchors)([input, pred_yolo_1, pred_yolo_2, pred_yolo_3]))
    }


def _make_yolo3_tiny_model(
        input_shape,
        max_box_per_image,
        nb_class,
        anchors,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        yolo_loss_options,
        debug_loss,
        model_scale_coefficient=1):
    input = Input(shape=input_shape, name='input_1')
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_2')
    # grid_h, grid_w, number of box in one anchor, 5+nb_class
    true_yolo_1 = Input(shape=(None, None, 3, 4 + 1 + nb_class), name='input_3')
    true_yolo_2 = Input(shape=(None, None, 3, 4 + 1 + nb_class), name='input_4')

    x = _conv_block(input, 1,
                    [{'filter': 16, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0}],
                    do_skip=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = _conv_block(x, 1,
                    [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 1}],
                    do_skip=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = _conv_block(x, 1,
                    [{'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2}],
                    do_skip=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = _conv_block(x, 1,
                    [{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}],
                    do_skip=False)
    x = skip = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = _conv_block(x, 1,
                    [{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 4}],
                    do_skip=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = _conv_block(x, 1,
                    [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 5}],
                    do_skip=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(x)

    last = _conv_block(x, 1,
                       [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6}],
                       do_skip=False)

    # Feature Pyramid Network with 2 heads
    # First branch
    pred_yolo_1 = _conv_block(last, 1,
                              [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7},
                               {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 8},
                               {'filter': (3 * (5 + nb_class)), 'no_scale': None, 'kernel': 1, 'stride': 1,
                                'bnorm': False, 'leaky': False, 'layer_idx': 9}],
                              do_skip=False)
    loss_yolo_1 = YoloLayer(anchors[6:],
                            [1 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            {**yolo_loss_options, 'grid_scale': yolo_loss_options['grid_scales'][0]},
                            debug_loss) \
        ([input, pred_yolo_1, true_yolo_1, true_boxes])

    # Second branch
    x = _conv_block(last, 1,
                    [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}],
                    do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip])
    pred_yolo_2 = _conv_block(x, 1,
                              [{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 11},
                               {'filter': (3 * (5 + nb_class)), 'no_scale': None, 'kernel': 1, 'stride': 1,
                                'bnorm': False, 'leaky': False, 'layer_idx': 12}],
                              do_skip=False)
    loss_yolo_2 = YoloLayer(anchors[:6],
                            [1 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            {**yolo_loss_options, 'grid_scale': yolo_loss_options['grid_scales'][0]},
                            debug_loss) \
        ([input, pred_yolo_2, true_yolo_2, true_boxes])

    return {
        'base_model': Model(input, last),
        'losses': [loss_yolo_1, loss_yolo_2],
        'train_model': Model([input, true_boxes, true_yolo_1, true_yolo_2],
                             [loss_yolo_1, loss_yolo_2]),
        'infer_model': Model(input, RegressionLayer(anchors)([input, pred_yolo_1, pred_yolo_2]))
    }


def _make_mobilenet_v2_model(
        input_shape,
        max_box_per_image,
        nb_class,
        anchors,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        yolo_loss_options,
        debug_loss,
        model_scale_coefficient=1):
    model = MobileNetV2(input_shape, alpha=model_scale_coefficient, include_top=False, weights=None)
    input, skip, last = model.layers[0].output, UpSampling2D(2)(model.get_layer('block_14_add').output),\
                        model.layers[-1].output

    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_2')
    # grid_h, grid_w, number of box in one anchor, 5+nb_class
    true_yolo_1 = Input(shape=(None, None, 3, 4 + 1 + nb_class), name='input_3')
    true_yolo_2 = Input(shape=(None, None, 3, 4 + 1 + nb_class), name='input_4')

    # Feature Pyramid Network with 2 heads
    # First branch
    pred_yolo_1 = _conv_block(last, 1.,
                              [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7},
                               {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 8},
                               {'filter': (3 * (5 + nb_class)), 'no_scale': None, 'kernel': 1, 'stride': 1,
                                'bnorm': False, 'leaky': False, 'layer_idx': 9}],
                              do_skip=False)
    loss_yolo_1 = YoloLayer(anchors[6:],
                            [1 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            {**yolo_loss_options, 'grid_scale': yolo_loss_options['grid_scales'][0]},
                            debug_loss) \
        ([input, pred_yolo_1, true_yolo_1, true_boxes])

    # Second branch
    x = _conv_block(last, 1,
                    [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}],
                    do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip])
    pred_yolo_2 = _conv_block(x, 1,
                              [{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 11},
                               {'filter': (3 * (5 + nb_class)), 'no_scale': None, 'kernel': 1, 'stride': 1,
                                'bnorm': False, 'leaky': False, 'layer_idx': 12}],
                              do_skip=False)
    loss_yolo_2 = YoloLayer(anchors[:6],
                            [1 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            {**yolo_loss_options, 'grid_scale': yolo_loss_options['grid_scales'][0]},
                            debug_loss) \
        ([input, pred_yolo_2, true_yolo_2, true_boxes])

    return {
        'base_model': Model(input, last),
        'losses': [loss_yolo_1, loss_yolo_2],
        'train_model': Model([input, true_boxes, true_yolo_1, true_yolo_2],
                             [loss_yolo_1, loss_yolo_2]),
        'infer_model': Model(input, RegressionLayer(anchors)([input, pred_yolo_1, pred_yolo_2]))
    }


def create_full_model(
        model_type,
        freeze_base_model,
        nb_class,
        anchors,
        max_box_per_image,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        yolo_loss_options,
        model_scale_coefficient,
        debug_loss
):
    assert(model_type in ['mobilenet2', 'tiny_yolo3', 'yolo3'])

    input_shape = None, None, 3

    models = None
    if model_type == 'mobilenet2':
        models = _make_mobilenet_v2_model(input_shape, max_box_per_image, nb_class, anchors, max_grid, batch_size,
                                          warmup_batches, ignore_thresh, yolo_loss_options, debug_loss,
                                          model_scale_coefficient)
    elif model_type == 'tiny_yolo3':
        models = _make_yolo3_tiny_model(input_shape, max_box_per_image, nb_class, anchors, max_grid, batch_size,
                                        warmup_batches, ignore_thresh, yolo_loss_options, debug_loss,
                                        model_scale_coefficient)
    elif model_type == 'yolo3':
        models = _make_yolo3_model(input_shape, max_box_per_image, nb_class, anchors, max_grid, batch_size,
                                   warmup_batches, ignore_thresh, yolo_loss_options, debug_loss,
                                   model_scale_coefficient)

    if freeze_base_model:
        for layer in models['base_model'].layers:
            layer.trainable = False

    return [models['train_model'], models['infer_model']]


def dummy_loss(y_true, y_pred):
    return y_pred
