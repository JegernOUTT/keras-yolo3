import os
import random
from typing import Tuple, List

import cv2
import numpy as np
from keras.utils import Sequence
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from scipy.misc import imresize


class MasknetBatchGenerator(Sequence):
    def __init__(self,
                 coco: COCO,
                 coco_dir: str,
                 classes: List[str],
                 input_image_size: Tuple[int, int],
                 mask_size: Tuple[int, int],
                 batch_size=1,
                 shuffle=True,
                 augmentation=True):
        self.coco = coco
        self.coco_dir = coco_dir
        self.image_ids = []
        self.classes = classes
        self.image_h, self.image_w = input_image_size
        self.mask_h, self.mask_w = mask_size
        self.batch_size = batch_size
        self.max_box_size = 20
        self.shuffle = shuffle
        self.augmentation = augmentation

        self._load_coco()

        if shuffle:
            np.random.shuffle(self.image_ids)

    def _load_coco(self):
        image_ids = self.coco.getImgIds(catIds=[1])  # in coco "person" category has 0 id

        for i in image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=i)
            anns = self.coco.loadAnns(ids=ann_ids)
            if all(['dp_masks' in ann.keys() for ann in anns]):
                self.image_ids.append(i)

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self):
            r_bound = len(self)
            l_bound = r_bound - self.batch_size

        image_batch = np.zeros((r_bound - l_bound, self.image_h, self.image_w, 3), dtype=np.float32)
        boxes_batch = np.zeros((r_bound - l_bound, self.max_box_size, 4), dtype=np.float32)
        masks_batch = np.zeros((r_bound - l_bound, self.max_box_size, self.mask_h, self.mask_w, len(self.classes)), dtype=np.float32)

        for i, image_id in enumerate(self.image_ids[l_bound:r_bound]):
            img = self.coco.loadImgs(ids=image_id)[0]
            img_filename, w, h = img['file_name'], img['width'], img['height']
            image_data = cv2.imread(os.path.join(self.coco_dir, img_filename))
            image_data = image_data[:, :, ::-1]
            image_data = cv2.resize(image_data, (self.image_w, self.image_h))
            # if self.augmentation:
            #     image_data = self._aug(image_data)
            image_batch[i, ...] = image_data / 255.
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(ids=ann_ids)

            for j, ann in enumerate(annotations):
                assert 'dp_masks' in ann and 'bbox' in ann

                x1, y1, x2, y2 = ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]
                boxes_batch[i, j, ...] = np.array([y1 / h, x1 / w, y2 / h, x2 / w], dtype=np.float32)
                for c in range(len(self.classes)):
                    if len(ann['dp_masks'][c]) == 0:
                        continue
                    mask = imresize(mask_util.decode(ann['dp_masks'][c]), (self.mask_h, self.mask_w), 'nearest')
                    mask = mask * 255. / 255.
                    masks_batch[i, j, ..., c] = mask

        return [image_batch, boxes_batch], [masks_batch]

    def _aug(self, image):
        image_h, image_w, _ = image.shape

        methods = [("area", cv2.INTER_AREA),
                   ("nearest", cv2.INTER_NEAREST),
                   ("linear", cv2.INTER_LINEAR),
                   ("cubic", cv2.INTER_CUBIC),
                   ("lanczos4", cv2.INTER_LANCZOS4)]
        image = cv2.resize(image, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)

        return image

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)


if __name__ == '__main__':
    batch_size = 1
    colors = [(255, 0, 0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (192,192,192),
              (128,128,128), (128,0,0), (128,128,0), (0,128,0), (128,0,128), (0,128,128),
              (0, 206, 209), (220,20,60)]
    categories = ['Torso', 'Right Hand', 'Left Hand', 'Left Foot', 'Right Foot', 'Upper Leg Right', 'Upper Leg Left',
                  'Lower Leg Right', 'Lower Leg Left', 'Upper Arm Left', 'Upper Arm Right', 'Lower Arm Left',
                  'Lower Arm Right', 'Head']
    coco_dir = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/coco_2014/'
    # train_coco = COCO(os.path.join(coco_dir, "annotations/densepose_coco_2014_train.json"))
    val_coco = COCO(os.path.join(coco_dir, "annotations/densepose_coco_2014_minival.json"))

    generator = MasknetBatchGenerator(coco=val_coco,
                                      coco_dir=os.path.join(coco_dir, 'images/val2014'),
                                      classes=categories,
                                      input_image_size=(416, 416),
                                      mask_size=(128, 128),
                                      batch_size=batch_size)

    for i in range(len(generator)):
        (image_batch, boxes_batch), masks_batch = generator[i]
        for b in range(batch_size):
            image, boxes, masks = image_batch[b], boxes_batch[b], masks_batch[b]
            image = (image * 255.).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h, w, _ = image.shape

            for j, box in enumerate(boxes):
                if np.all(np.isclose(box, 0.)):
                    continue
                y1, x1, y2, x2 = box
                y1, x1, y2, x2 = int(np.trunc(y1 * h)), int(np.trunc(x1 * w)), int(np.trunc(y2 * h)), int(np.trunc(x2 * w))
                curr_masks = np.transpose(masks[b, j], (2, 0, 1))
                for mask_idx, mask in enumerate(curr_masks):
                    if np.all(np.isclose(mask, 0.)):
                        continue
                    mask = imresize(mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
                    mask = np.where(mask >= 0.5, 1, 0).astype(np.uint8)
                    color = colors[mask_idx]
                    mask = cv2.merge((mask * color[0], mask * color[1], mask * color[2]))
                    image[y1:y2, x1:x2] = cv2.addWeighted(image[y1:y2, x1:x2], 1.0, mask, 0.8, 0)

            cv2.imshow('image', image)
            key = cv2.waitKeyEx(0)
            if key == 27:
                break
        if key == 27:
            break
