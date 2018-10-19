import cv2

import imgaug as ia
from imgaug import augmenters as iaa

from preprocessing import *


def _is_on_the_edge(bb, w, h):
    return np.isclose(bb.x1, 0.) or np.isclose(bb.x2, 0.) or np.isclose(bb.y1, 0.) or np.isclose(bb.y2, 0.) \
           or np.isclose(bb.x1, w) or np.isclose(bb.x2, w) or np.isclose(bb.y1, h) or np.isclose(bb.y2, h)


def _is_edge_big_diff(bb):
    w = bb.x2 - bb.x1
    h = bb.y2 - bb.y1
    return (max(w, h) / min(w, h)) > 4.


def _is_area_small(bb, w, h, area_limit=40.):
    bb_w = bb.x2 - bb.x1
    bb_h = bb.y2 - bb.y1
    return np.sqrt((w * h) / (bb_w * bb_h)) > area_limit


def _is_area_out_of_image_large(bb, w, h, max_area_ratio=1.):
    top_bb_area = max(0., (bb.x2 - bb.x1) * (0. - bb.y1))
    left_bb_area = max(0., (0. - bb.x1) * (bb.y2 - bb.y1))
    right_bb_area = max(0., (bb.x2 - w) * (bb.y2 - bb.y1))
    down_bb_area = max(0., (bb.x2 - bb.x1) * (bb.y2 - h))

    cutten_box_area = (min(bb.x2, w) - max(bb.x1, 0.)) * (min(bb.y2, h) - max(bb.y1, 0.))

    return cutten_box_area / (
            top_bb_area + left_bb_area + right_bb_area + down_bb_area + np.finfo(np.float).eps) < max_area_ratio


def fill_filtered_with_gray(img, bbs):
    h, w, _ = img.shape
    for bb in bbs.bounding_boxes:
        img[int(max(bb.y1, 0)):int(min(bb.y2, h)), int(max(bb.x1, 0)):int(min(bb.x2, w))] = 128


def remove_on_the_edge(bbs, image_shape):
    h, w, _ = image_shape

    filtered = []

    tmp_bbs = []
    for bb in bbs.bounding_boxes:
        if not _is_area_out_of_image_large(bb, w, h):
            tmp_bbs.append(bb)
        else:
            filtered.append(bb)

    final_bbs = []
    for bb in ia.BoundingBoxesOnImage(tmp_bbs, shape=image_shape).cut_out_of_image().bounding_boxes:
        if not ((_is_on_the_edge(bb, w, h) and _is_edge_big_diff(bb))\
                or (_is_on_the_edge(bb, w, h) and _is_area_small(bb, w, h))):
            final_bbs.append(bb)
        else:
            filtered.append(bb)

    return ia.BoundingBoxesOnImage(final_bbs, shape=image_shape), \
           ia.BoundingBoxesOnImage(filtered, shape=image_shape).cut_out_of_image()


class Augment:
    def __init__(self):
        self.seq = iaa.Sequential(
            [
                iaa.Crop(percent=(0.0, 0.01)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.01),
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                iaa.ContrastNormalization((0.5, 1.5)),
                iaa.Affine(shear=(-5, 5), cval=128)
            ])

    def __call__(self, instance, annotations, need_to_augment=True):
        image = cv2.imread(instance['file_name'])
        if image is None:
            print(instance['file_name'])
        image = image[:, :, ::-1]
        if not need_to_augment:
            return image, annotations

        h, w, _ = image.shape
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=bbox['xmin'], y1=bbox['ymin'], x2=bbox['xmax'], y2=bbox['ymax'],
                           label=bbox)
            for bbox in annotations
            if bbox['xmin'] < bbox['xmax'] and bbox['ymin'] < bbox['ymax']
        ], shape=image.shape)

        seq_det = self.seq.to_deterministic()

        img_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0].remove_out_of_image()
        bbs_aug, filtered_bbs = remove_on_the_edge(bbs_aug, img_aug.shape)
        fill_filtered_with_gray(img_aug, filtered_bbs)

        return img_aug, [{
            'name': bb.label['name'],
            'category_id': bb.label['category_id'],
            'xmin': bb.x1,
            'xmax': bb.x2,
            'ymin': bb.y1,
            'ymax': bb.y2
        } for bb in bbs_aug.bounding_boxes]
