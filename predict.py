#! /usr/bin/env python
import datetime
import os
import argparse
import json
import cv2
import keras
from keras_applications.mobilenet_v2 import relu6

from utils.utils import get_yolo_boxes, add_regression_layer_if_not_exists
from utils.bbox import draw_boxes
from keras.models import load_model

from yolo import RegressionLayer

model_name = '{}_model.h5'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def is_tiny_model(model_name):
    return model_name in ['tiny_yolo3', 'mobilenet2']


def _main_(args):
    keras.backend.set_learning_phase(0)
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
        model_config = config['model']
        config = config['inference']

    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    is_tiny = is_tiny_model(model_config['type'])
    net_size = config['input_size']
    anchors = model_config['anchors'] if not is_tiny else model_config['tiny_anchors']
    obj_thresh, nms_thresh = config['obj_thresh'], config['nms_thresh']
    snapshot_name = os.path.join(config["snapshots_path"], model_name.format(model_config['type']))
    input_path = config["input_path"]
    every_nth = config["process_nth_frame"]

    if not os.path.exists(snapshot_name):
        raise FileNotFoundError(snapshot_name)

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    custom_objects = {'RegressionLayer': RegressionLayer}
    if config['is_mobilenet2']:
        custom_objects = {'relu6': relu6}
    infer_model = load_model(snapshot_name, custom_objects=custom_objects)
    infer_model = add_regression_layer_if_not_exists(infer_model, anchors)

    video_reader = cv2.VideoCapture(input_path)

    if config['need_to_save_output']:
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_writer = cv2.VideoWriter(config['output_path'], cv2.VideoWriter_fourcc(*'MPEG'),
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

        orig_image = image.copy()
        boxes = get_yolo_boxes(infer_model, image, net_size, net_size, obj_thresh, nms_thresh)
        image = draw_boxes(image, boxes, model_config['labels'], obj_thresh)

        if config['need_to_save_output']:
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

        # save image
        elif key == 83 or key == 115:  # s
            if not os.path.exists('images'):
                os.mkdir('images')
            cv2.imwrite('images/{}.jpg'.format(datetime.datetime.now()), orig_image)

        # net_size
        elif key == 90 or key == 122:  # z
            net_size = net_size if net_size <= 64 else net_size - 32
        elif key == 67 or key == 99:  # c
            net_size = net_size if net_size >= 1600 else net_size + 32

        print('\rProcessed {} frame: net_size[{}]; obj_thresh[{:.2f}]; nms_thresh[{:.2f}]; process_nth_frame[{}]'.format(
            frames_processed, net_size, obj_thresh, nms_thresh, every_nth
        ), end='')

    if config['need_to_save_output']:
        video_writer.release()
    video_reader.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Predict with a trained model')

    argparser.add_argument(
        '-c',
        '--conf',
        default='config.json',
        help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
