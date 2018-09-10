import importlib
import os
import keras.backend as K

import cv2
import numpy as np
from keras.engine.saving import load_model

from classifier.classifier_utils import get_yolo3_output, draw_boxes
from classifier.model import create_classifier_model
from yolo import RegressionLayer


if __name__ == '__main__':
    K.set_learning_phase(0)

    config = importlib.import_module('config').config
    os.environ['CUDA_VISIBLE_DEVICES'] = config['predict']['cuda_devices']
    batch_size = config['predict']['classifier_batch_size']

    infer_model = load_model(config['predict']['infer_model_path'], custom_objects={'RegressionLayer': RegressionLayer})
    C2 = infer_model.get_layer('leaky_10').output
    C3 = infer_model.get_layer('leaky_35').output
    C4 = infer_model.get_layer('leaky_60').output
    C5 = infer_model.get_layer('leaky_78').output

    classifier_model = create_classifier_model(
        image_shape=(config['predict']['classifier_net_size'], config['predict']['classifier_net_size'], 3),
        classes_count=len(config['categories']))
    assert os.path.exists(config['predict']['classifier_weights_path'])
    classifier_model.load_weights(config['predict']['classifier_weights_path'])
    classifier_model.summary()

    video_reader = cv2.VideoCapture(config['predict']['videofile_path'])

    frames_processed = 0
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    while True:
        read, image = video_reader.read()
        if not read:
            break
        orig_image = image.copy()

        frames_processed += 1
        if frames_processed % config['predict']['every_nth_frame'] != 0:
            continue

        boxes, crops = get_yolo3_output(
            infer_model, image,
            config['predict']['infer_net_size'],
            config['predict']['classifier_net_size'],
            config['predict']['confidence'],
            config['predict']['nms_threshold'])
        if len(crops) == 0:
            cv2.imshow('image', image)
            cv2.waitKey(1)
            continue

        h, w, _ = image.shape

        classes = np.zeros((len(crops), len(config['categories'])))
        for i in range(len(crops) // batch_size):
            out = classifier_model.predict_on_batch(crops[i * batch_size: (i + 1) * batch_size, ...])
            for j in range(len(out)):
                classes[i + j] = out[j]

        boxes = [[b.xmin, b.ymin, b.xmax, b.ymax, np.max(classes[i]), 1 if np.isclose(classes[i, np.argmax(classes[i])], 0.) else np.argmax(classes[i])]
                 for i, b in enumerate(boxes)]
        image = draw_boxes(image, boxes, config['categories'])
        cv2.imshow('image', image)
        key = cv2.waitKeyEx(1)
        if key == 27:  # esc
            break

        # process_nth_frame
        elif key == 81 or key == 113:  # q
            config['predict']['every_nth_frame'] = config['predict']['every_nth_frame'] \
                if config['predict']['every_nth_frame'] <= 1 else config['predict']['every_nth_frame'] - 1
        elif key == 69 or key == 101:  # e
            config['predict']['every_nth_frame'] = config['predict']['every_nth_frame'] \
                if config['predict']['every_nth_frame'] >= 100 else config['predict']['every_nth_frame'] + 1

        # net_size
        elif key == 90 or key == 122:  # z
            config['predict']['infer_net_size'] = config['predict']['infer_net_size'] \
                if config['predict']['infer_net_size'] <= 64 else config['predict']['infer_net_size'] - 32
        elif key == 67 or key == 99:  # c
            config['predict']['infer_net_size'] = config['predict']['infer_net_size'] \
                if config['predict']['infer_net_size'] >= 1600 else config['predict']['infer_net_size'] + 32

        # obj_thresh
        elif key == 64 or key == 97:  # a
            config['predict']['confidence'] = config['predict']['confidence'] \
                if config['predict']['confidence'] <= 0.1 else config['predict']['confidence'] - 0.05
        elif key == 68 or key == 100:  # d
            config['predict']['confidence'] = config['predict']['confidence'] \
                if config['predict']['confidence'] >= 0.95 else config['predict']['confidence'] + 0.05

        print('\rProcessed {} frame: net_size[{}]; obj_thresh[{:.2f}]; nms_thresh[{:.2f}]; process_nth_frame[{}]'.format(
            frames_processed, config['predict']['infer_net_size'], config['predict']['confidence'],
            config['predict']['nms_threshold'], config['predict']['every_nth_frame']
        ), end='')
    cv2.destroyAllWindows()
