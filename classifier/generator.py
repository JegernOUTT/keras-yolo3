import importlib
import random

import cv2

import numpy as np
from keras.utils import Sequence, to_categorical

from classifier.classifier_utils import get_normalized_image, draw_boxes
from preprocessing import TrassirRectShapesAnnotations
from utils.image import correct_bounding_boxes, random_flip, apply_random_scale_and_crop


class ClassifierBatchGenerator(Sequence):
    def __init__(self,
                 instances,
                 classes,
                 net_image_size,
                 max_box_per_image=200,
                 batch_size=1,
                 shuffle=True,
                 augmentation=True):
        self.instances = instances
        self.classes = classes
        self.net_h, self.net_w = net_image_size
        self.batch_size = batch_size
        self.max_box_per_image = max_box_per_image
        self.shuffle = shuffle
        self.augmentation = augmentation

        if shuffle:
            np.random.shuffle(instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances)) / self.batch_size))

    def __getitem__(self, idx):
        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        image_batch = np.zeros((r_bound - l_bound, self.net_h, self.net_w, 3))
        boxes_batch = np.zeros((r_bound - l_bound, self.max_box_per_image, 4))
        classes_batch = np.zeros((r_bound - l_bound, self.max_box_per_image, len(self.classes)))

        for idx, train_instance in enumerate(self.instances[l_bound:r_bound]):
            image = get_normalized_image(train_instance['file_name'])

            annotations = [{
                'category_id': a['category_id'],
                'xmin': a['bbox'][0][0],
                'xmax': a['bbox'][1][0],
                'ymin': a['bbox'][0][1],
                'ymax': a['bbox'][1][1]
            } for a in train_instance['annotations']]

            if self.augmentation:
                image, annotations = self._aug(image, annotations)

            image_batch[idx] = image
            for b_i, b in enumerate(annotations):
                boxes_batch[idx][b_i] = np.array([b['ymin'], b['xmin'], b['ymax'], b['xmax']], dtype=np.float32)
                classes_batch[idx][b_i] = to_categorical([b['category_id']], num_classes=len(self.classes))

        return [image_batch, boxes_batch], classes_batch

    def _aug(self, image, annotations, jitter=0.):
        image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        dw = jitter * image_w
        dh = jitter * image_h

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(0.25, 2) if not np.isclose(jitter, 0.0) else 1.

        if new_ar < 1:
            new_h = int(scale * self.net_h)
            new_w = int(self.net_h * new_ar)
        else:
            new_w = int(scale * self.net_w)
            new_h = int(self.net_w / new_ar)

        dx = int(np.random.uniform(0, self.net_w - new_w) * jitter)
        dy = int(np.random.uniform(0, self.net_h - new_h) * jitter)

        # apply scaling and cropping
        methods = [("area", cv2.INTER_AREA),
                   ("nearest", cv2.INTER_NEAREST),
                   ("linear", cv2.INTER_LINEAR),
                   ("cubic", cv2.INTER_CUBIC),
                   ("lanczos4", cv2.INTER_LANCZOS4)]
        image = cv2.resize(image, (new_w, new_h), interpolation=random.choice(methods)[1])
        image, paddings = apply_random_scale_and_crop(image, new_w, new_h, self.net_w, self.net_h, dx, dy)
        if paddings:  # no jitter
            dx, dy = paddings

        # randomly flip
        flip = np.random.randint(2) if not np.isclose(jitter, 0.0) else 0
        image = random_flip(image, flip)

        # correct the size and pos of bounding boxes
        annotations = [{
            'category_id': a['category_id'],
            'xmin': int(round(a['xmin'] * image_w)) if int(round(a['xmin'] * image_w)) < image_w else image_w,
            'xmax': int(round(a['xmax'] * image_w)) if int(round(a['xmax'] * image_w)) < image_w else image_w,
            'ymin': int(round(a['ymin'] * image_h)) if int(round(a['ymin'] * image_h)) < image_h else image_h,
            'ymax': int(round(a['ymax'] * image_h)) if int(round(a['ymax'] * image_h)) < image_h else image_h
        } for a in annotations]
        all_objs = correct_bounding_boxes(annotations,
                                          new_w, new_h,
                                          self.net_w, self.net_h,
                                          dx, dy,
                                          flip,
                                          image_w, image_h)

        return image, [{
            'category_id': a['category_id'],
            'xmin': a['xmin'] / self.net_w,
            'xmax': a['xmax'] / self.net_w,
            'ymin': a['ymin'] / self.net_h,
            'ymax': a['ymax'] / self.net_h
        } for a in all_objs]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.instances)


if __name__ == '__main__':
    config = importlib.import_module('config').config
    net_size = config['eval']['net_size']
    annotations = TrassirRectShapesAnnotations(
        config['datasets']['train'] + config['datasets']['val'],
        [],
        categories=config['categories'],
        skip_categories=[])
    annotations.load()
    annotations.print_statistics()

    data = annotations.get_train_instances(verifiers=config['verifiers'], max_bbox_per_image=200)
    train_generator = ClassifierBatchGenerator(data, config['categories'], (net_size, net_size), batch_size=1)

    for i in range(len(train_generator)):
        (image_batch, boxes_batch), classes_batch = train_generator[i]
        image, boxes, classes = image_batch[0], boxes_batch[0], classes_batch[0]
        image = (image * 255.).astype(np.uint8)

        boxes = [[b[1], b[0], b[3], b[2], 1., np.argmax(classes[i])] for i, b in enumerate(boxes)]
        image = draw_boxes(image, boxes, config['categories'])

        cv2.imshow('image', cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        key = cv2.waitKeyEx(0) & 0xFF
        if key == 27:
            break
