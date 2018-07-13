#! /usr/bin/env python

import numpy as np
import os
import argparse
import json
import cv2
import keras
from keras import Model, Input
from keras_applications.mobilenet_v2 import relu6
from scipy.misc import imresize

import masknet
from utils.utils import preprocess_input, decode_netout, correct_yolo_boxes, do_nms
from utils.bbox import draw_boxes
from keras.models import load_model

model_name = '{}_model.h5'
masknet_model_name = '{}_masknet_model.h5'
masknet_weights_name = '{}_masknet_weights.h5'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def get_yolo3_output(model, image, net_h, net_w, anchors, obj_thresh, nms_thresh):
    image_h, image_w, _ = image.shape
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
    keras.backend.set_learning_phase(0)

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
        masknet_config = config['masknet_model']
        model_config = config['model']
        infer_config = config['inference']

    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    net_size = masknet_config['input_size']
    obj_thresh, nms_thresh = infer_config['obj_thresh'], infer_config['nms_thresh']
    snapshot_name = os.path.join(masknet_config["snapshots_path"], model_name.format(model_config['type']))
    input_path = infer_config["input_path"]
    every_nth = infer_config["process_nth_frame"]

    if not os.path.exists(snapshot_name):
        raise FileNotFoundError(snapshot_name)

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    custom_objects = {}
    if infer_config['is_mobilenet2']:
        custom_objects = {'relu6': relu6}
    infer_model = load_model(snapshot_name, custom_objects=custom_objects)

    C2 = infer_model.get_layer('leaky_10').output
    C3 = infer_model.get_layer('leaky_35').output
    C4 = infer_model.get_layer('leaky_60').output
    C5 = infer_model.get_layer('leaky_78').output

    mn_model = masknet.create_model((net_size, net_size, 3))
    m_roi_input = Input(shape=(None, 4), name='input_2')
    x = mn_model([C2, C3, C4, C5, m_roi_input])
    model = Model(inputs=[infer_model.inputs[0], m_roi_input], outputs=x)
    model.summary()

    masknet_snapshot_weights_name = os.path.join(masknet_config["snapshots_path"],
                                                 masknet_weights_name.format(model_config['type']))
    mn_model.load_weights(masknet_snapshot_weights_name)

    video_reader = cv2.VideoCapture(input_path)
    if infer_config['need_to_save_output']:
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_writer = cv2.VideoWriter(infer_config['output_path'], cv2.VideoWriter_fourcc(*'MPEG'),
                                       50.0, (frame_w, frame_h))
    frames_processed = 0
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    while True:
        read, image = video_reader.read()
        if not read:
            break

        frames_processed += 1

        if frames_processed % every_nth != 0:
            continue

        w, h, _ = image.shape

        # predict the bounding boxes
        boxes, masknet_outputs = get_yolo3_output(infer_model, image, net_size, net_size, model_config['anchors'],
                                                  obj_thresh, nms_thresh)
        valid_boxes = correct_yolo_boxes(boxes, h, w, net_size, net_size)
        image = draw_boxes(image, valid_boxes, model_config['labels'], obj_thresh)

        boxes = [b for b in boxes if b.classes[0] > obj_thresh]
        rois = np.array([[b.ymin, b.xmin, b.ymax, b.xmax] for b in boxes], dtype=np.float32)

        C2, C3, C4, C5 = masknet_outputs

        p = mn_model.predict([C2, C3, C4, C5, np.expand_dims(rois, axis=0)])
        p = p[0, :len(boxes), :, :, 0]
        if (float(net_size) / w) < (float(net_size) / h):
            new_w = net_size
            new_h = (h * net_size) / w
        else:
            new_h = net_size
            new_w = (w * net_size) / h
        for i in range(len(boxes)):
            roi = rois[i]
            mask = p[i]
            y1, x1, y2, x2 = roi

            x_offset, x_scale = (net_size - new_w) / 2. / net_size, float(new_w) / net_size
            y_offset, y_scale = (net_size - new_h) / 2. / net_size, float(new_h) / net_size

            left = min(max(int((x1 - x_offset) / x_scale * w), 0), w)
            top = min(max(int((y1 - y_offset) / y_scale * h), 0), h)
            right = min(max(int((x2 - x_offset) / x_scale * w), 0), w)
            bot = min(max(int((y2 - y_offset) / y_scale * h), 0), h)

            mask = imresize(mask, (bot - top, right - left), interp='bilinear').astype(np.float32) / 255.0
            mask2 = np.where(mask >= 0.5, 1, 0).astype(np.uint8)
            if (i % 3) == 0:
                mask3 = cv2.merge((mask2 * 0, mask2 * 0, mask2 * 255))
            elif (i % 3) == 1:
                mask3 = cv2.merge((mask2 * 0, mask2 * 255, mask2 * 0))
            else:
                mask3 = cv2.merge((mask2 * 255, mask2 * 0, mask2 * 0))

            image[top:bot, left:right] = cv2.addWeighted(image[top:bot, left:right], 1.0, mask3, 0.8, 0)

        if infer_config['need_to_save_output']:
            video_writer.write(image)

        cv2.imshow('image', image)
        key = cv2.waitKeyEx(1)
        if key == 27:  # esc
            break
        elif key == 32:  # space
            while cv2.waitKeyEx(0) != 32:
                pass
        # process_nth_frame
        elif key == 81 or key == 113:  # q
            every_nth = every_nth if every_nth <= 1 else every_nth - 1
        elif key == 69 or key == 101:  # e
            every_nth = every_nth if every_nth >= 100 else every_nth + 1

        # obj_thresh
        elif key == 64 or key == 97:  # a
            obj_thresh = obj_thresh if obj_thresh <= 0.1 else obj_thresh - 0.05
        elif key == 68 or key == 100:  # d
            obj_thresh = obj_thresh if obj_thresh >= 0.95 else obj_thresh + 0.05

        print('\rProcessed {} frame: net_size[{}]; obj_thresh[{:.2f}]; nms_thresh[{:.2f}]; process_nth_frame[{}]'.format(
            frames_processed, net_size, obj_thresh, nms_thresh, every_nth
        ), end='')

    if infer_config['need_to_save_output']:
        video_writer.release()
    video_reader.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Predict with a masknet model')

    argparser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
