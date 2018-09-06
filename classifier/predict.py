import importlib
import os
import keras.backend as K

import cv2
import numpy as np
from keras.engine.saving import load_model
from keras.layers import Input
from keras.models import Model

from classifier.classifier_utils import get_yolo3_output_with_features, draw_boxes
from classifier.model import create_classifier_model
from yolo import RegressionLayer


if __name__ == '__main__':
    K.set_learning_phase(0)

    config = importlib.import_module('config').config
    os.environ['CUDA_VISIBLE_DEVICES'] = config['predict']['cuda_devices']

    infer_model = load_model(config['infer_model_path'], custom_objects={'RegressionLayer': RegressionLayer})
    C2 = infer_model.get_layer('leaky_10').output
    C3 = infer_model.get_layer('leaky_35').output
    C4 = infer_model.get_layer('leaky_60').output
    C5 = infer_model.get_layer('leaky_78').output

    classifier_model = create_classifier_model(
        image_size=(config['predict']['net_size'], config['predict']['net_size'], 3),
        classes_count=len(config['categories']))
    assert os.path.exists(config['predict']['classifier_weights_path'])
    classifier_model.load_weights(config['predict']['classifier_weights_path'])
    classifier_model.summary()

    m_roi_input = Input(shape=(None, 4), name='input_2')
    x = classifier_model([C2, C3, C4, C5, m_roi_input])

    common_model = Model(inputs=[infer_model.inputs[0], m_roi_input], outputs=x)
    common_model.summary()

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

        masknet_output, boxes = get_yolo3_output_with_features(
            infer_model, image,
            config['predict']['net_size'], config['predict']['net_size'],
            config['predict']['confidence'], config['predict']['nms_threshold'])
        if len(boxes) == 0:
            cv2.imshow('image', image)
            cv2.waitKey(1)
            continue

        h, w, _ = image.shape
        rois = np.array([[b.ymin / h, b.xmin / w, b.ymax / h, b.xmax / w] for b in boxes], dtype=np.float32)
        classes = classifier_model.predict([*masknet_output, np.expand_dims(rois, axis=0)])
        classes = classes[0]

        boxes = [[b.xmin, b.ymin, b.xmax, b.ymax, np.max(classes[i]), np.argmax(classes[i])]
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
            config['predict']['net_size'] = config['predict']['net_size'] \
                if config['predict']['net_size'] <= 64 else config['predict']['net_size'] - 32
        elif key == 67 or key == 99:  # c
            config['predict']['net_size'] = config['predict']['net_size'] \
                if config['predict']['net_size'] >= 1600 else config['predict']['net_size'] + 32

        # obj_thresh
        elif key == 64 or key == 97:  # a
            config['predict']['confidence'] = config['predict']['confidence'] \
                if config['predict']['confidence'] <= 0.1 else config['predict']['confidence'] - 0.05
        elif key == 68 or key == 100:  # d
            config['predict']['confidence'] = config['predict']['confidence'] \
                if config['predict']['confidence'] >= 0.95 else config['predict']['confidence'] + 0.05

        print('\rProcessed {} frame: net_size[{}]; obj_thresh[{:.2f}]; nms_thresh[{:.2f}]; process_nth_frame[{}]'.format(
            frames_processed, config['predict']['net_size'], config['predict']['confidence'],
            config['predict']['nms_threshold'], config['predict']['every_nth_frame']
        ), end='')
    cv2.destroyAllWindows()
