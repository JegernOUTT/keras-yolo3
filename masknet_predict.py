#! /usr/bin/env python
import math

import numpy as np
import os
import argparse
import json
import cv2
import keras
from keras import Model, Input
from scipy.misc import imresize

import masknet
from utils.utils import preprocess_input, decode_netout, correct_yolo_boxes, do_nms
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # define the GPU to work on here


def get_yolo3_output(model, image, net_h, net_w, anchors, obj_thresh, nms_thresh):
    image_h, image_w, _ = image.shape
    # new_image = preprocess_input(image, net_h, net_w)
    new_image = preprocess_input(image, net_h, net_w)

    inp = model.input  # input placeholder
    layers_names = ['leaky_10', 'leaky_35', 'leaky_60', 'leaky_78',
                    'conv_81', 'conv_93', 'conv_105']
    outputs = [layer.output for layer in model.layers if layer.name in layers_names]
    functor = keras.backend.function([inp], outputs)

    layer_outs = functor([new_image])

    masknet_output, yolos = layer_outs[:4], layer_outs[4:]

    # Get boxes from yolo prediction
    boxes = []
    for i in range(len(yolos)):
        # decode the output of the network
        yolo_anchors = anchors[(2 - i) * 6:(3 - i) * 6]  # config['model']['anchors']
        boxes += decode_netout(yolos[i][0], yolo_anchors, obj_thresh, net_h, net_w)

    do_nms(boxes, nms_thresh)

    return boxes, masknet_output


def _main_(args):
    ###############################
    #   Set some parameter
    ###############################
    keras.backend.set_learning_phase(0)

    labels = ['person']
    anchors = [10,37, 17,71, 28,104, 28,50, 42,79, 45,148, 70,92, 77,181, 193,310]
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.8, 0.4

    ###############################
    #   Load the model
    ###############################
    infer_model = load_model('./snapshots/person_trassir/yolo3_model.h5')
    for layer in infer_model.layers:
        layer.trainable = False

    C2 = infer_model.get_layer('leaky_10').output
    C3 = infer_model.get_layer('leaky_35').output
    C4 = infer_model.get_layer('leaky_60').output
    C5 = infer_model.get_layer('leaky_78').output

    mn_model = masknet.create_model((net_h, net_w, 3))
    m_roi_input = Input(shape=(None, 4), name='input_2')
    x = mn_model([C2, C3, C4, C5, m_roi_input])
    model = Model(inputs=[infer_model.inputs[0], m_roi_input], outputs=x)
    model.summary()

    mn_model.load_weights('weights.hdf5')

    ###############################
    #   Predict bounding boxes
    ###############################

    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/Проход касса 16-17_20180327-112348--20180327-113348.tmp.avi'
    input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/nothing/3.avi'

    # do detection on a video
    video_reader = cv2.VideoCapture(input_path)
    every_nth = 1

    for i in tqdm(range(100000)):
        # Run online profiling with 'op' view and 'opts' options at step 15, 18, 20.

        _, image = video_reader.read()
        h, w, _ = image.shape

        if i % every_nth != 0:
            continue

        # predict the bounding boxes
        boxes, masknet_outputs = get_yolo3_output(infer_model, image, net_h, net_w, anchors,
                                                  obj_thresh, nms_thresh)
        valid_boxes = correct_yolo_boxes(boxes, h, w, net_h, net_w)
        image = draw_boxes(image, valid_boxes, labels, obj_thresh)

        boxes = [b for b in boxes if b.classes[0] > obj_thresh]
        rois = [[b.ymin, b.xmin, b.ymax, b.xmax] for b in boxes]
        rois = np.array(rois, dtype=np.float32)

        C2, C3, C4, C5 = masknet_outputs

        p = mn_model.predict([C2, C3, C4, C5, np.expand_dims(rois, axis=0)])
        p = p[0, :len(boxes), :, :, 0]
        if (float(net_w) / w) < (float(net_h) / h):
            new_w = net_w
            new_h = (h * net_w) / w
        else:
            new_h = net_w
            new_w = (w * net_h) / h
        for i in range(len(boxes)):
            roi = rois[i]
            mask = p[i]
            y1, x1, y2, x2 = roi

            x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
            y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

            left = int((x1 - x_offset) / x_scale * w)
            right = int((x2 - x_offset) / x_scale * w)
            top = int((y1 - y_offset) / y_scale * h)
            bot = int((y2 - y_offset) / y_scale * h)

            left = min(max(left, 0), w)
            top = min(max(top, 0), h)
            right = min(max(right, 0), w)
            bot = min(max(bot, 0), h)

            mask = imresize(mask, (bot - top, right - left), interp='bilinear').astype(np.float32) / 255.0
            mask2 = np.where(mask >= 0.5, 1, 0).astype(np.uint8)
            if (i % 3) == 0:
                mask3 = cv2.merge((mask2 * 0, mask2 * 0, mask2 * 255))
            elif (i % 3) == 1:
                mask3 = cv2.merge((mask2 * 0, mask2 * 255, mask2 * 0))
            else:
                mask3 = cv2.merge((mask2 * 255, mask2 * 0, mask2 * 0))

            image[top:bot, left:right] = cv2.addWeighted(image[top:bot, left:right], 1.0, mask3, 0.8, 0)

        cv2.imshow('Image', cv2.resize(image, (1280, 720)))
        cv2.waitKey(1)

    video_reader.release()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Predict with a trained yolo model')

    argparser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')

    argparser.add_argument(
        '-i',
        '--input',
        help='path to an image or an video (mp4 format)')

    args = argparser.parse_args()
    _main_(args)
