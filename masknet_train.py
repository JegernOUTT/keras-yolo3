#!/usr/bin/python3
import os
import random
import threading
from random import shuffle

import cv2
import keras
import keras.backend as K
import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Input
from keras.metrics import binary_accuracy
from keras.models import Model
from pycocotools.coco import COCO
from scipy.misc import imresize

import masknet


def roi_pool_cpu(frame, bbox, pool_size):
    frame_h, frame_w = frame.shape[:2]

    x1 = int(bbox[0] * frame_w)
    y1 = int(bbox[1] * frame_h)
    w1 = int(bbox[2] * frame_w)
    h1 = int(bbox[3] * frame_h)

    slc = frame[y1:y1+h1,x1:x1+w1,...]

    if (w1 <= 0) or (h1 <= 0):
        assert(np.count_nonzero(slc) == 0)
        return slc

    slc = imresize(slc.astype(float), (pool_size, pool_size), 'nearest') / 255.0

    return slc


# TODO: convert to our own annotations format
def process_coco(coco, img_path, limit):
    res = []
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(ids = img_ids)
    processed = 0
    iter1 = 0

    if limit:
        imgs = imgs[:limit]

    for img in imgs:
        iter1 += 1
        processed += 1
        if iter1 > 1000:
            iter1 = 0
            print("processed", processed, len(imgs))

        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)
        frame_w = img['width']
        frame_h = img['height']
        rois = []
        msks = []
        bboxs = []
        cocos = []
        for ann in anns:
            if ('bbox' in ann) and (ann['bbox'] != []) and ('segmentation' in ann):
                bbox = [int(xx) for xx in ann['bbox']]
                bbox[0] /= frame_w
                bbox[1] /= frame_h
                bbox[2] /= frame_w
                bbox[3] /= frame_h

                m = coco.annToMask(ann)

                if m.max() < 1:
                    continue

                if ann['iscrowd']:
                    continue

                msk = roi_pool_cpu(m, bbox, masknet.my_msk_inp * 2)

                if np.count_nonzero(msk) == 0:
                    continue

                assert(len(rois) < masknet.my_num_rois)

                x1 = np.float32(bbox[0])
                y1 = np.float32(bbox[1])
                w1 = np.float32(bbox[2])
                h1 = np.float32(bbox[3])

                rois.append([y1, x1, y1 + h1, x1 + w1])
                msks.append(ann)
                bboxs.append(bbox)
                cocos.append(coco)

        if len(rois) > 0:
            for _ in range(masknet.my_num_rois - len(rois)):
                rois.append([np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)])
                msks.append(None)
                bboxs.append(None)
                cocos.append(None)
            res.append((img['file_name'], img_path, rois, msks, bboxs, cocos))

    return res


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def my_preprocess(im, h, w):
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return imsz


@threadsafe_generator
def fit_generator(imgs, batch_size, net_h, net_w):
    ii = 0
    fake_msk = np.zeros((masknet.my_msk_inp * 2, masknet.my_msk_inp * 2), dtype=np.uint8).astype('float32')
    while True:
        shuffle(imgs)
        for k in range(len(imgs) // batch_size):
            i = k * batch_size
            j = i + batch_size
            if j > len(imgs):
                j = - j % len(imgs)
            batch = imgs[i:j]
            x1 = []
            x2 = []
            y = []
            for img_name, img_path, rois, anns, bboxs, cocos in batch:
                flip = random.randint(0, 1) == 0
                # flip = False

                frame = cv2.imread(os.path.join(img_path, img_name))
                if flip:
                    frame = np.fliplr(frame)
                x1.append(my_preprocess(frame, net_h, net_w))

                my_rois = []
                for roi in rois:
                    rx1 = roi[1]
                    rx2 = roi[3]
                    if flip:
                        rx1 = 1.0 - roi[3]
                        rx2 = 1.0 - roi[1]
                    my_rois.append([roi[0], rx1, roi[2], rx2])

                x2.append(np.array(my_rois))

                msks = []
                for k in range(len(bboxs)):
                    if cocos[k] is None:
                        msk = fake_msk
                    else:
                        msk = roi_pool_cpu(cocos[k].annToMask(anns[k]), bboxs[k], masknet.my_msk_inp * 2)
                        if flip:
                            msk = np.fliplr(msk)
                    msks.append(msk)

                msks = np.array(msks)
                msks = msks[..., np.newaxis]

                y.append(msks)
            #gc.collect()
            #print("yield",ii)
            ii += 1
            yield ([np.array(x1), np.array(x2)], np.array(y))


def my_accuracy(y_true, y_pred):
    mask_shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))
    mask_shape = tf.shape(y_true)
    y_true = K.reshape(y_true, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))

    sm = tf.reduce_sum(y_true, [1, 2, 3])

    ix = tf.where(sm > 0)[:, 0]

    y_true = tf.gather(y_true, ix)
    y_pred = tf.gather(y_pred, ix)

    return binary_accuracy(y_true, y_pred)


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__(
            filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        tmp_model = self.model
        self.model = self.my_model
        super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.model = tmp_model


if __name__ == "__main__":
    net_h, net_w = 416, 416

    infer_model = load_model('./snapshots/person_trassir/yolo3_model.h5')
    for layer in infer_model.layers:
        layer.trainable = False

    C2 = infer_model.get_layer('leaky_10').output
    C3 = infer_model.get_layer('leaky_35').output
    C4 = infer_model.get_layer('leaky_60').output
    C5 = infer_model.get_layer('leaky_78').output

    mn_model = masknet.create_model(image_size=(net_h, net_w, 3))
    mn_model.summary()

    m_roi_input = Input(shape=(masknet.my_num_rois, 4), name='input_2')

    x = mn_model([C2, C3, C4, C5, m_roi_input])

    model = Model(inputs=[infer_model.inputs[0], m_roi_input], outputs=x)
    model.summary()
    model.compile(loss=[masknet.my_loss], optimizer='adam', metrics=[my_accuracy])
    mn_model.load_weights("weights.hdf5")

    bdir = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/coco'
    train_coco = COCO(os.path.join(bdir, "annotations/person_keypoints_train2017.json"))
    val_coco = COCO(os.path.join(bdir, "annotations/person_keypoints_val2017.json"))
    train_imgs = process_coco(train_coco, os.path.join(bdir, "images/train2017"), None)
    val_imgs = process_coco(val_coco, os.path.join(bdir, "images/val2017"), None)

    train_coco = None
    val_coco = None

    train_imgs += val_imgs[5000:]
    val_imgs = val_imgs[:5000]

    batch_size = 64

    train_data = fit_generator(train_imgs, batch_size, net_h, net_w)

    validation_data = fit_generator(val_imgs, batch_size, net_h, net_w)

    lr_schedule = lambda epoch: 0.001 if epoch < 20 else 0.0001
    callbacks = [LearningRateScheduler(lr_schedule)]

    mcp = MyModelCheckpoint(filepath="weights.hdf5", monitor='val_loss', save_best_only=True)
    mcp.my_model = mn_model

    callbacks.append(mcp)

    callbacks.append(ModelCheckpoint(filepath="all_weights.hdf5", monitor='val_loss', save_best_only=True))

    model.fit_generator(train_data,
                        steps_per_epoch=len(train_imgs) / batch_size,
                        validation_steps=len(val_imgs) / batch_size,
                        epochs=100,
                        validation_data=validation_data,
                        max_queue_size=10,
                        workers=2,
                        use_multiprocessing=False,
                        verbose=1,
                        callbacks=callbacks)
