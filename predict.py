#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # define the GPU to work on here

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


def _main_(args):
    config_path = args.conf
    input_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    ###############################

    # do detection on a video
    video_reader = cv2.VideoCapture(input_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(nb_frames)):
        _, image = video_reader.read()

        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, image, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh)
        cv2.imshow('Image', image)
        cv2.waitKey(1)

    video_reader.release()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
