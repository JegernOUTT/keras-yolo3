import warnings

import cv2

import keras
import tensorflow as tf
import numpy as np

from utils.bbox import draw_boxes
from utils.utils import get_yolo_boxes, evaluate

from keras.callbacks import TensorBoard, ModelCheckpoint


class EvaluateCallback(keras.callbacks.Callback):
    def __init__(self, infer_model, anchors, labels, video_path):
        super().__init__()
        self.anchors = anchors
        self.labels = labels
        self.video_path = video_path
        self.infer_model = infer_model
        self.net_h, self.net_w = 416, 416
        self.obj_thresh, self.nms_thresh = 0.5, 0.1
        self.every_n_frame, self.max_frame_count = 50, 5

    def on_epoch_end(self, batch, logs={}):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        obj_thresh, nms_thresh = 0.5, 0.1

        video_reader = cv2.VideoCapture(self.video_path)
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


class CustomTensorBoard(TensorBoard):
    def __init__(self, infer_model, validation_generator, **kwargs):
        super().__init__(**kwargs)
        self.infer_model = infer_model
        self.validation_generator = validation_generator
        self.counter = 0

    def on_epoch_end(self, batch, logs={}):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.counter)

        recalls, average_precisions = evaluate(self.infer_model, self.validation_generator)

        mAP = np.mean(list(average_precisions.values()))
        recall = np.mean(list(recalls.values()))

        print('mAP: {}, recall: {}'.format(mAP, recall))

        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = mAP
        summary_value.tag = 'mAP'
        self.writer.add_summary(summary, self.counter)

        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = recall
        summary_value.tag = 'recall'
        self.writer.add_summary(summary, self.counter)

        self.writer.flush()
        self.counter += 1

        super(CustomTensorBoard, self).on_epoch_end(batch, logs)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, model_to_save, **kwargs):
        super(CustomModelCheckpoint, self).__init__(**kwargs)
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)

        super(CustomModelCheckpoint, self).on_batch_end(epoch, logs)
