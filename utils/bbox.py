import hashlib

import numpy as np
from PIL import ImageDraw, Image


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / max(float(union), 0.00000001)


def generate_color_by_text(text):
    hash_code = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return b, g, r, 100


def draw_boxes(image, boxes, labels, obj_thresh):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image, "RGBA")

    for box in filter(lambda x: x.get_score() > obj_thresh, boxes):
        label = box.get_label()
        label_str = labels[label]

        if label >= 0:
            draw.text((box.xmin + 5, box.ymin + 5), label_str)
            draw.rectangle([box.xmin, box.ymin, box.xmax, box.ymax],
                           fill=generate_color_by_text(label_str),
                           outline=(255, 255, 255, 255))

    return np.array(image.convert('RGB'))
