import copy
import os
from typing import List, Tuple

import cv2
import numpy as np
import keras
from PIL import Image, ImageDraw

from utils.bbox import generate_color_by_text
from utils.utils import preprocess_input, decode_netout, do_nms, correct_yolo_boxes


def get_normalized_image(filename: str) -> np.array:
    assert os.path.exists(filename)
    image = cv2.imread(filename)
    image = image[:, :, ::-1]
    return image / 255.


def draw_boxes(image: np.array,
               boxes: list,
               classes: List[str]) -> np.array:
    h, w, _ = image.shape
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image, "RGBA")

    for box in boxes:
        xmin, ymin, xmax, ymax, conf, class_id = box
        normalized = 0. <= xmin <= 1.
        if normalized:
            xmin, ymin, xmax, ymax = xmin * w, ymin * h, xmax * w, ymax * h
        draw.text((xmin + 5, ymin + 5), '{} {:.2f}'.format(classes[class_id], conf))
        draw.rectangle([xmin, ymin, xmax, ymax],
                       fill=generate_color_by_text(classes[class_id]),
                       outline=(255, 255, 255, 255))

    return np.array(image.convert('RGB'))


def get_yolo3_output(model: keras.Model,
                     image: np.array,
                     infer_net_size: int,
                     classifier_net_size: int,
                     obj_thresh: float,
                     nms_thresh: float) -> Tuple[list, np.array]:
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, infer_net_size, infer_net_size)
    output = model.predict(new_image)

    # Get boxes from yolo prediction
    boxes = decode_netout(output, obj_thresh)
    boxes = correct_yolo_boxes(boxes, image_h, image_w, infer_net_size, infer_net_size)
    do_nms(boxes, nms_thresh)

    boxes = [b for b in boxes if b.classes[0] > obj_thresh]

    crops = np.zeros((len(boxes), classifier_net_size, classifier_net_size, 3))
    for i, b in enumerate(boxes):
        crop = image[b.ymin:b.ymax, b.xmin:b.xmax, :] / 255.
        try:
            crop = cv2.resize(crop, (classifier_net_size, classifier_net_size), cv2.INTER_LANCZOS4)
        except cv2.error:
            continue
        crops[i] = crop

    return boxes, crops


