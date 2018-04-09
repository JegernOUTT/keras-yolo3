import json
import os
import pickle
from copy import deepcopy

import numpy as np


def load_images(config, skip_empty=True):
    images_dir = config['train']['images_dir']

    train_last_image_index = 0
    train_data = {
        'images_with_annotations': [],
        'categories': []
    }

    # Train dataset loading
    for dataset in config['train']['datasets_to_train']:
        current_path = os.path.join(images_dir, dataset['path'])

        if not skip_empty and not (os.path.isfile(os.path.join(current_path, 'annotations.pickle')) or
                                   os.path.isfile(os.path.join(current_path, 'annotations.json'))):
            assert False, "Error path: {}".format(os.path.join(current_path, 'annotations.pickle'))

        if os.path.isfile(os.path.join(current_path, 'annotations.pickle')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = pickle.load(f)
        elif os.path.isfile(os.path.join(current_path, 'annotations.json')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = json.load(f)

        assert annotations

        if dataset['only_verified']:
            annotations['images'] = [image for image in annotations['images']
                                     if any(map(lambda x: x, image['verified'].values()))]

        for image in annotations['images']:
            image['file_name'] = os.path.join(images_dir, dataset['path'], image['file_name'])

        if dataset['count_to_process'] != "all":
            np.random.shuffle(annotations['images'])
            annotations['images'] = annotations['images'][:int(dataset['count_to_process'])]

        if len(train_data['categories']) == 0:
            train_data['categories'] = annotations['categories']
        elif len(annotations['categories']) == 0:
            pass
        else:
            current_categories = set(map(lambda x: (x['id'], x['name']), annotations['categories']))
            for category in train_data['categories']:
                cat = (category['id'], category['name'])
                assert cat in current_categories, 'Categories ids must be same in all datasets'

        image_id_to_image = {image['id']: image for image in annotations['images']}
        images_with_annotations = {image['id']: [] for image in annotations['images']}
        for annotation in annotations['annotations']:
            if annotation['image_id'] not in image_id_to_image:
                continue
            image = image_id_to_image[annotation['image_id']]
            image_area = image['width'] * image['height']
            bbox_area = (annotation['bbox'][1][0] * image['width'] - annotation['bbox'][0][0] * image['width']) * \
                        (annotation['bbox'][1][1] * image['height'] - annotation['bbox'][0][1] * image['height'])
            area_ratio = bbox_area / image_area

            if area_ratio < dataset['min_bbox_area'] or area_ratio > dataset['max_bbox_area']:
                continue

            images_with_annotations[annotation['image_id']].append(annotation)

        for image_id, anns in images_with_annotations.items():
            image, anns = deepcopy(image_id_to_image[image_id]), deepcopy(anns)
            image['id'] = train_last_image_index
            for annotation in anns:
                annotation['image_id'] = train_last_image_index
            train_data['images_with_annotations'].append((image, anns))
            train_last_image_index += 1

    val_last_image_index = 0
    validation_data = {
        'images_with_annotations': [],
        'categories': []
    }

    # Validation dataset loading
    for dataset in config['train']['datasets_to_validate']:
        current_path = os.path.join(images_dir, dataset['path'])
        if not skip_empty and not (os.path.isfile(os.path.join(current_path, 'annotations.pickle')) or
                                   os.path.isfile(os.path.join(current_path, 'annotations.json'))):
            assert False, "Error path: {}".format(os.path.join(current_path, 'annotations.pickle'))

        if os.path.isfile(os.path.join(current_path, 'annotations.pickle')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = pickle.load(f)
        elif os.path.isfile(os.path.join(current_path, 'annotations.json')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = json.load(f)

        assert annotations

        if dataset['only_verified']:
            annotations['images'] = [image for image in annotations['images']
                                     if any(map(lambda x: x, image['verified'].values()))]

        for image in annotations['images']:
            image['file_name'] = os.path.join(images_dir, dataset['path'], image['file_name'])

        if dataset['count_to_process'] != "all":
            np.random.shuffle(annotations['images'])
            annotations['images'] = annotations['images'][:int(dataset['count_to_process'])]

        if len(validation_data['categories']) == 0:
            validation_data['categories'] = annotations['categories']
        elif len(annotations['categories']) == 0:
            pass
        else:
            current_categories = set(map(lambda x: (x['id'], x['name']), annotations['categories']))
            for category in train_data['categories']:
                cat = (category['id'], category['name'])
                assert cat in current_categories, 'Categories ids must be same in all datasets'

        image_id_to_image = {image['id']: image for image in annotations['images']}
        images_with_annotations = {image['id']: [] for image in annotations['images']}
        for annotation in annotations['annotations']:
            if annotation['image_id'] not in image_id_to_image:
                continue
            image = image_id_to_image[annotation['image_id']]
            image_area = image['width'] * image['height']
            bbox_area = (annotation['bbox'][1][0] * image['width'] - annotation['bbox'][0][0] * image['width']) * \
                        (annotation['bbox'][1][1] * image['height'] - annotation['bbox'][0][1] * image['height'])

            if dataset['min_bbox_area'] > bbox_area / image_area > dataset['max_bbox_area']:
                continue

            images_with_annotations[annotation['image_id']].append(annotation)

        for image_id, anns in images_with_annotations.items():
            image, anns = deepcopy(image_id_to_image[image_id]), deepcopy(anns)
            image['id'] = val_last_image_index
            for annotation in anns:
                annotation['image_id'] = val_last_image_index
            validation_data['images_with_annotations'].append((image, anns))
            val_last_image_index += 1

    return train_data, validation_data
