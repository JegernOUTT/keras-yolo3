# !/usr/bin/python3

import keras.layers as KL
from keras.layers import Input
from keras.models import Model

from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


def resize_to(input_tensor, dim_out):
    return tf.image.resize_nearest_neighbor(input_tensor, [dim_out[0], dim_out[1]])


class PyramidROIAlign(Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]
    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, image_shape, pool_shape=(14, 14), **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    @staticmethod
    def log2_graph(x):
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
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

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
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1], )


class RoiPoolingConv(Layer):
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert(self.dim_ordering == 'tf')

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        rois_flattened = K.reshape(rois, (-1, 4))

        shape = tf.shape(rois_flattened)
        box_indices = tf.range(0, shape[0]) // self.num_rois

        res = tf.image.crop_and_resize(
            img, rois_flattened, box_indices, (self.pool_size, self.pool_size),
            method="bilinear")

        res = K.reshape(res, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return res


class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


def create_model(image_size):
    assert len(image_size) == 3

    C2 = Input(shape=(None, None, 128), name="input_C2")
    C3 = Input(shape=(None, None, 256), name="input_C3")
    C4 = Input(shape=(None, None, 512), name="input_C4")
    C5 = Input(shape=(None, None, 1024), name="input_C5")

    roi_input = Input(shape=(None, 4), name="input_rois")

    P5 = KL.Conv2D(64, (1, 1))(C5)
    P4 = KL.Add()([
        KL.UpSampling2D(size=(2, 2))(P5),
        KL.Conv2D(64, (1, 1))(C4)
    ])
    P3 = KL.Add()([
        KL.UpSampling2D(size=(2, 2))(P4),
        KL.Conv2D(64, (1, 1))(C3)
    ])
    P2 = KL.Add()([
        KL.UpSampling2D(size=(2, 2))(P3),
        KL.Conv2D(64, (1, 1))(C2)
    ])

    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(64, (3, 3), padding="same")(P2)
    P3 = KL.Conv2D(64, (3, 3), padding="same")(P3)
    P4 = KL.Conv2D(64, (3, 3), padding="same")(P4)
    P5 = KL.Conv2D(64, (3, 3), padding="same")(P5)

    feature_maps = [P2, P3, P4, P5]

    roi_pool_layer = PyramidROIAlign(image_size, name="roi_align")([roi_input] + feature_maps)

    for _ in range(8):
        x = KL.TimeDistributed(KL.Conv2D(512, (4, 4), padding="same"))(roi_pool_layer)
        x = KL.TimeDistributed(BatchNorm(axis=3))(x)
        x = KL.Activation('relu')(x)

    class_idx = KL.TimeDistributed(KL.Conv2DTranspose(512, (2, 2), strides=2, activation="relu"))(x)
    u_coord = KL.TimeDistributed(KL.Conv2DTranspose(512, (2, 2), strides=2, activation="relu"))(x)
    v_coord = KL.TimeDistributed(KL.Conv2DTranspose(512, (2, 2), strides=2, activation="relu"))(x)

    class_idx = KL.TimeDistributed(KL.Lambda(resize_to, arguments={'dim_out': (25, 25)}, name='class_idx'))(class_idx)
    u_coord = KL.TimeDistributed(KL.Lambda(resize_to, arguments={'dim_out': (25, 25)}, name='u_coord'))(u_coord)
    v_coord = KL.TimeDistributed(KL.Lambda(resize_to, arguments={'dim_out': (25, 25)}, name='v_coord'))(v_coord)

    return Model(inputs=[C2, C3, C4, C5, roi_input], outputs=[class_idx, u_coord, v_coord])


if __name__ == '__main__':
    model = create_model((512, 512, 3))
    model.summary()