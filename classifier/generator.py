import importlib
import os
import random
import shutil

import cv2

import numpy as np
from keras.utils import Sequence, to_categorical

from classifier.classifier_utils import get_normalized_image, draw_boxes
from preprocessing import TrassirRectShapesAnnotations


class ClassifierBatchGenerator(Sequence):
    def __init__(self,
                 instances,
                 classes,
                 net_image_size,
                 is_train,
                 batch_size=1,
                 shuffle=True,
                 augmentation=True):
        self.instances = instances
        self.files = []
        self.classes = classes
        self.net_h, self.net_w = net_image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation

        if is_train:
            self.tmp_path = '.tmp_train_crops'
        else:
            self.tmp_path = '.tmp_val_crops'
        if os.path.exists(self.tmp_path):
            shutil.rmtree(self.tmp_path)
        os.mkdir(self.tmp_path)
        
        self.create_crops()

        if shuffle:
            np.random.shuffle(self.files)

    def __del__(self):
        if os.path.exists(self.tmp_path):
            shutil.rmtree(self.tmp_path)

    def create_crops(self):
        for item in self.instances:
            image = cv2.imread(item['file_name'])
            h, w, _ = image.shape

            for i, b in enumerate(item['annotations']):
                if b['category_name'] not in self.classes:
                    continue

                x1, y1, x2, y2 = b['bbox'][0][0], b['bbox'][0][1], b['bbox'][1][0], b['bbox'][1][1]
                b_w = x2 - x1
                
                x1 = max(min(int((x1 - (b_w / 2)) * w), w), 0)
                y1 = max(min(int((y1 - (b_w / 2)) * h), h), 0)
                x2 = max(min(int((x2 + (b_w / 2)) * w), w), 0)
                y2 = max(min(int((y2 + (b_w / 2)) * h), h), 0)

                if x1 == x2 or y1 == y2:
                    continue

                assert x1 < x2 and y1 < y2

                filename = '{}__{}_{}.jpg'.format(b['category_name'], os.path.basename(item['file_name']), i)
                filename = os.path.join(self.tmp_path, filename)

                cv2.imwrite(filename, image[y1:y2, x1:x2, :])
                self.files.append(filename)

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self):
            r_bound = len(self)
            l_bound = r_bound - self.batch_size

        image_batch = np.zeros((r_bound - l_bound, self.net_h, self.net_w, 3), dtype=np.float32)
        classes_batch = np.zeros((r_bound - l_bound, len(self.classes)), dtype=np.float32)

        for idx, file_name in enumerate(self.files[l_bound:r_bound]):
            class_name = os.path.basename(file_name).split('__')[0]
            image = get_normalized_image(file_name)

            if self.augmentation:
                image = self._aug(image)

            image_batch[idx] = image
            classes_batch[idx] = to_categorical(self.classes.index(class_name), num_classes=len(self.classes))

        return image_batch, classes_batch

    def _aug(self, image):
            image_h, image_w, _ = image.shape

            # apply scaling and cropping
            methods = [("area", cv2.INTER_AREA),
                       ("nearest", cv2.INTER_NEAREST),
                       ("linear", cv2.INTER_LINEAR),
                       ("cubic", cv2.INTER_CUBIC),
                       ("lanczos4", cv2.INTER_LANCZOS4)]
            image = cv2.resize(image, (self.net_w, self.net_h), interpolation=random.choice(methods)[1])

            return image

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)


if __name__ == '__main__':
    config = importlib.import_module('config').config
    net_size = config['eval']['net_size']
    batch_size = 16
    annotations = TrassirRectShapesAnnotations(
        config['datasets']['train'] + config['datasets']['val'],
        [],
        categories=config['categories'],
        skip_categories=[])
    annotations.load()
    annotations.print_statistics()

    data = annotations.get_train_instances(verifiers=config['verifiers'], max_bbox_per_image=200)
    train_generator = ClassifierBatchGenerator(data, config['categories'], (net_size, net_size),
                                               batch_size=batch_size, is_train=False)

    for i in range(len(train_generator)):
        image_batch, classes_batch = train_generator[i]
        for b in range(batch_size):
            image, classes = image_batch[b], classes_batch[b]
            image = (image * 255.).astype(np.uint8)

            cv2.putText(image,
                        config['categories'][np.argmax(classes)],
                        (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 0), 2)

            cv2.imshow('image', image[:, :, ::-1])
            key = cv2.waitKeyEx(0) & 0xFF
            if key == 27:
                break
