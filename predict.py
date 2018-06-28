#! /usr/bin/env python

import os
import argparse
import json
import cv2
import keras
from keras.applications import mobilenet

from utils.utils import get_yolo_boxes
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # define the GPU to work on here


def _main_(args):
    config_path = args.conf
    input_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    # os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.8, 0.3

    ###############################
    #   Load the model
    ###############################
    infer_model = load_model('./snapshots/current_person/yolo3_model.h5')
    # infer_model = load_model('./snapshots/carface/yolo3_model.h5')

    ###############################
    #   Predict bounding boxes
    ###############################

    input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/Проход касса 16-17_20180327-112348--20180327-113348.tmp.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/nothing/3.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/nothing/1.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues-hard/Очередь 3_20150323-174453--20150323-181951.tva.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/left/0.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/left/5.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues-hard/касса 2-3_nzvsm_2.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/test/DP-kass-5ka/DS-2CD2542FWD-IS 3_20180419-180000--20180419-190000-1.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/test/DP-kass-5ka/DS-2CD2542FWD-IS 3_20180419-180000--20180419-190000-2.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/test/DP-kass-5ka/DS-2CD2542FWD-IS 3_20180419-180000--20180419-190000-3.avi'
    # input_path = 'rtsp://admin:hik12345@172.16.16.34/Streaming/Channels/1'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/Downloads/1.mp4'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/Downloads/2.mp4'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/Downloads/3.mp4'
    # input_path = 'rtsp://172.17.17.54:555/qRbD2KXT_m/'
    # input_path = 'rtsp://admin:admin1337@172.16.17.13'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/кассы 8-9_20171110-192101--20171110-192601.avi'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/TR-D2123IR3v2 4_20180604-090709--20180604-091328.avi'
    # input_path = 'rtsp://192.168.0.35:555/S1uxjhff_m/'
    # input_path = 'rtsp://192.168.0.35:555/VmSjBQbX_m/'
    # input_path = 'rtsp://172.20.25.144:555/FOJuplwr_m/'
    # input_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/Румыны/2018-06-07_063105.avi'
    # input_path = '/mnt/nfs/Data/LPR/new_video/ro/ro500.avi'
    # input_path = '/mnt/nfs/Data/LPR/new_video/kz/new_yellow/drive-download-20180528T085002Z-001.avi'
    # input_path = '/home/svakhreev/workspace/t4/chroots/trassir_gentoo/sys/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/305214949504/ArabesqueTimisoara-7.avi'
    input_path = '/home/svakhreev/workspace/t4/chroots/trassir_gentoo/sys/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/305214949504/ArabesqueTimisoara-5.avi'
    # input_path = '/home/svakhreev/workspace/t4/chroots/trassir_gentoo/sys/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/305214949504/ArabesqueTimisoara-14.avi'

    # do detection on a video
    video_reader = cv2.VideoCapture(input_path)
    # nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    # video_writer = cv2.VideoWriter('/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/china_out.avi',
    #                                cv2.VideoWriter_fourcc(*'MPEG'),
    #                                50.0,
    #                                (frame_w, frame_h))
    every_nth = 10

    for i in tqdm(range(100000)):
        # Run online profiling with 'op' view and 'opts' options at step 15, 18, 20.

        _, image = video_reader.read()

        if i % every_nth != 0:
            continue

        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, image, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

        # draw bounding boxes on the image using labels
        image = draw_boxes(image, boxes, config['model']['labels'], obj_thresh)
        # video_writer.write(image)
        cv2.imshow('Image', cv2.resize(image, (1280, 720)))
        cv2.waitKey(1)

    # video_writer.release()
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
