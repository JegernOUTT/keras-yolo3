import datetime
import math
import tqdm

import cv2
import numpy as np
from .bbox import BoundBox, bbox_iou


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def evaluate(model,
             generator,
             iou_threshold=0.5,
             obj_thresh=0.5,
             nms_thresh=0.3,
             net_h=608,
             net_w=608,
             batch_size=32):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in np.arange(0, generator.size(), batch_size):
        if i + batch_size > generator.size():
            batch_size = generator.size() - i
        raw_images = np.array([generator.load_image(j) for j in range(i, i + batch_size)])
        pred_boxes = get_yolo_boxes_batch(model, raw_images, net_h, net_w,
                                          generator.get_anchors(), obj_thresh, nms_thresh)
        # pred_boxes = [[box for box in pred if box.get_score() > obj_thresh] for pred in pred_boxes]

        for shifted_i in range(i, i + batch_size):
            score = np.array([box.get_score() for box in pred_boxes[shifted_i - i]])
            pred_labels = np.array([box.label for box in pred_boxes[shifted_i - i]])

            if len(pred_boxes[shifted_i - i]) > 0:
                current_pred_boxes = np.array([[int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax), box.get_score()]
                                               for box in pred_boxes[shifted_i - i]])
            else:
                current_pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            current_pred_boxes = current_pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[shifted_i][label] = current_pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(shifted_i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                if len(np.squeeze(annotations)) > 0:
                    all_annotations[shifted_i][label] = annotations[annotations[:, 4] == label, :4].copy()
                else:
                    all_annotations[shifted_i][label] = np.array([])

    # compute mAP by comparing all detections and all annotations
    average_precisions = {}
    recalls = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        recalls[label] = np.max(recall) if len(recall) else 0
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return recalls, average_precisions


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    valid_boxes = []
    is_valid = lambda x: math.isfinite(x) and not math.isnan(x)
    for box in boxes:
        if not is_valid(box.xmin) or not is_valid(box.xmax) or not is_valid(box.ymin) or not is_valid(box.ymax):
            continue
        box.xmin = box.xmin if box.xmin > 0 else 0
        box.ymin = box.ymin if box.ymin > 0 else 0
        box.xmax = box.xmax if box.xmax < 1.0 else 1.0
        box.ymax = box.ymax if box.ymax < 1.0 else 1.0
        valid_boxes.append(box)

    for i in range(len(valid_boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        valid_boxes[i].xmin = int((valid_boxes[i].xmin - x_offset) / x_scale * image_w)
        valid_boxes[i].xmax = int((valid_boxes[i].xmax - x_offset) / x_scale * image_w)
        valid_boxes[i].ymin = int((valid_boxes[i].ymin - y_offset) / y_scale * image_h)
        valid_boxes[i].ymax = int((valid_boxes[i].ymax - y_offset) / y_scale * image_h)

    return valid_boxes


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

 
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if objectness <= obj_thresh:
                continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row, col, b, :4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[row, col, b, 5:]

            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)

            boxes.append(box)

    return boxes


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h), cv2.INTER_LANCZOS4)

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def preprocess_input_batch(images, net_h, net_w):
    preprocessed_images = np.zeros((images.shape[0], net_h, net_w, 3), dtype=np.float)
    for i, image in enumerate(images):
        new_h, new_w, _ = image.shape

        # determine the new size of the image
        if (float(net_w) / new_w) < (float(net_h) / new_h):
            new_h = (new_h * net_w) // new_w
            new_w = net_w
        else:
            new_w = (new_w * net_h) // new_h
            new_h = net_h

        # resize the image to the new size
        resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h), cv2.INTER_LANCZOS4)

        # embed the image into the standard letter box
        new_image = np.ones((net_h, net_w, 3)) * 0.5
        new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
        preprocessed_images[i] = new_image

    return preprocessed_images


def normalize(image):
    return image / 255.


def get_yolo_boxes(model, image, net_h, net_w, anchors, obj_thresh, nms_thresh):
    # preprocess the input
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)

    # run the prediction
    yolos = model.predict(new_image)
    boxes = []

    for i in range(len(yolos)):
        # decode the output of the network
        yolo_anchors = anchors[(2 - i) * 6:(3 - i) * 6]  # config['model']['anchors']
        boxes += decode_netout(yolos[i][0], yolo_anchors, obj_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    boxes = correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)

    return boxes


def get_yolo_boxes_batch(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    # preprocess the input
    new_images = preprocess_input_batch(images, net_h, net_w)

    # run the prediction
    yolos = model.predict(new_images)

    batch_boxes = []
    for batch_i in range(len(images)):
        boxes = []
        image_h, image_w, _ = images[batch_i].shape

        for i in range(len(yolos)):
            # decode the output of the network
            yolo_anchors = anchors[(2 - i) * 6:(3 - i) * 6]  # config['model']['anchors']
            boxes += decode_netout(yolos[i][batch_i], yolo_anchors, obj_thresh, net_h, net_w)

        boxes = correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
        do_nms(boxes, nms_thresh)

        batch_boxes.append(boxes)

    return batch_boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
