import cv2

import keras

from utils.bbox import draw_boxes
from utils.utils import get_yolo_boxes


class EvaluateCallback(keras.callbacks.Callback):
    def __init__(self, infer_model, anchors, labels):
        super().__init__()
        self.anchors = anchors
        self.labels = labels
        self.model_path = '/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/' \
                          'очереди/Lanser 3MP-16 10_20171110-193448--20171110-194108.avi'
        self.infer_model = infer_model
        self.net_h, self.net_w = 416, 416
        self.obj_thresh, self.nms_thresh = 0.5, 0.45
        self.every_n_frame, self.max_frame_count = 50, 5

    def on_epoch_end(self, batch, logs={}):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        obj_thresh, nms_thresh = 0.5, 0.45

        video_reader = cv2.VideoCapture(self.model_path)
        current_frame_n = 0
        for i in range(int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, image = video_reader.read()
            if i % self.every_n_frame != 0:
                continue

            if current_frame_n > self.max_frame_count:
                break
            else:
                current_frame_n += 1

            boxes = get_yolo_boxes(self.infer_model,
                                   image,
                                   self.net_h,
                                   self.net_w,
                                   self.anchors,
                                   self.obj_thresh,
                                   self.nms_thresh)

            draw_boxes(image, boxes, self.labels, obj_thresh)
            cv2.imshow('image', image)
            cv2.waitKey(1000)
        video_reader.release()
        cv2.destroyWindow('image')
