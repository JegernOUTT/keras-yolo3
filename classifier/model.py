from typing import Tuple, Union

import keras
import keras.backend as K
import keras.layers as KL
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Input
from keras.models import Model


class PyramidROIAlign(Layer):
    def __init__(self, image_shape, pool_shape=(14, 14), **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.image_shape = tuple(image_shape)
        self.pool_shape = tuple(pool_shape)

    @staticmethod
    def log2_graph(x):
        """Implementatin of Log2. TF doesn't have a native implemenation."""
        return tf.log(x) / tf.log(2.0)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(
            self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            # level_boxes = tf.stop_gradient(level_boxes)
            # box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1],)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def create_classifier_model(image_size: Tuple[int, int, int],
                            classes_count: Union[None, int]) -> keras.Model:
    C2 = Input(shape=(None, None, 128), name="input_C2")
    C3 = Input(shape=(None, None, 256), name="input_C3")
    C4 = Input(shape=(None, None, 512), name="input_C4")
    C5 = Input(shape=(None, None, 1024), name="input_C5")

    roi_input = Input(shape=(None, 4), name="input_rois")

    P5 = KL.Conv2D(256, (1, 1), name='fpn_c5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)
    ])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)
    ])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)
    ])

    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(128, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(128, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(128, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(128, (3, 3), padding="SAME", name="fpn_p5")(P5)

    feature_maps = [P2, P3, P4, P5]

    roi_pool_layer = PyramidROIAlign(image_size, name="roi_align")([roi_input] + feature_maps)

    x = KL.TimeDistributed(KL.Conv2D(64, (3, 3), padding="same"))(roi_pool_layer)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3))(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.TimeDistributed(KL.Conv2D(64, (3, 3), padding="same"))(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3))(x)
    x = KL.LeakyReLU(alpha=0.3)(x)

    x = KL.TimeDistributed(KL.Conv2D(128, (3, 3), padding="same"))(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3))(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.TimeDistributed(KL.Conv2D(128, (3, 3), padding="same"))(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3))(x)
    x = KL.LeakyReLU(alpha=0.3)(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"))(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3))(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"))(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3))(x)
    x = KL.LeakyReLU(alpha=0.3)(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"))(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3))(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"))(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3))(x)
    x = KL.LeakyReLU(alpha=0.3)(x)

    x = KL.TimeDistributed(KL.GlobalAveragePooling2D())(x)
    x = KL.TimeDistributed(KL.Dense(classes_count, name="output", activation='softmax'))(x)

    return Model(inputs=[C2, C3, C4, C5, roi_input], outputs=x)
