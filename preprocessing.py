import json
import os
import pickle
import logging
import random
from operator import itemgetter

import numpy as np

from copy import deepcopy
from tqdm import tqdm


class TrassirRectShapesAnnotations:
    ANNOTATION_VERSION = '1.1'
    ANNOTATIONS_FILENAMES = [('annotations.pickle', pickle), ('annotations.json', json)]
    ANNOTATION_TYPE = 'RectShape'

    def __init__(self, train_folders, validation_folders):
        self._train_folders = train_folders
        self._validation_folders = validation_folders

        self._common_image_id = 0
        self._train_index = None
        self._validation_index = None
        self._categories = None
        self._verifiers = None

        # Extra indexes
        self._images_by_id_index = None
        self._categories_by_id_index = None
        self._categories_by_name_index = None

    def load(self):
        logging.info('Starting folders loading')
        self._load_all_folders()
        logging.info('All folders was loaded. Starting categories loading')
        self._retrieve_categories()
        logging.info('Categories was loaded. Retrieving verifiers')
        self._retrieve_verifiers()
        logging.info('Loading of annotations competed. Building indexes')
        self._build_extra_indexes()

    def print_statistics(self):
        if self._images_by_id_index is None or self._categories_by_id_index is None \
                or self._categories_by_name_index is None:
            logging.error('Before statistic retrieving you must to build index')
            return

        print('In training set {} images'.format(len(self._train_index)))
        print('In validation set {} images'.format(len(self._validation_index)))

        common_index = self._train_index + self._validation_index

        print('\nCategories:')
        for category in self._categories:
            print('[{}]: {}'.format(category['id'], category['name']))
        print('\nVerifiers:')
        for verifier, ids in self._verifiers.items():
            print('{}: {} images verified'.format(verifier, len(ids)))

        print('\nDetailed categories statistic:')
        categories_statistic = {c['id']: 0 for c in self._categories}
        for image in common_index:
            for ann in image['annotations']:
                categories_statistic[ann['category_id']] += 1
        for i, count in categories_statistic.items():
            print('[{}]: {} annotations'.format(self._categories_by_id_index[i]['name'], count))

    def get_train_instances(self, categories, verifiers, max_bbox_per_image, shuffle=True):
        final_images = []
        for folder in self._train_folders.values():
            images = folder['images']
            images = self._filter_by_image_availability(images)
            if folder['only_verified']:
                images = self._filter_by_verifiers(verifiers if verifiers != 'any' else None, images)
            if folder['images_count'] != "all":
                random.shuffle(images)
                images = images[:int(folder['images_count'])]

            for image in images:
                annotations = image['annotations']
                annotations = self._filter_by_categories(categories, annotations)
                annotations = self._filter_by_bbox_size(folder['bbox_limits'], (image['width'], image['height']),
                                                        annotations)
                annotations = self._filter_by_annotations_count(max_bbox_per_image, annotations)
                image['annotations'] = annotations
                final_images.append(image)
        if shuffle:
            random.shuffle(final_images)
        return final_images

    def get_validation_instances(self, categories, verifiers, max_bbox_per_image, shuffle=False):
        final_images = []
        for folder in self._validation_folders.values():
            images = folder['images']
            images = self._filter_by_image_availability(images)
            if folder['only_verified']:
                images = self._filter_by_verifiers(verifiers if verifiers != 'any' else None, images)
            if folder['images_count'] != "all":
                random.shuffle(images)
                images = images[:int(folder['images_count'])]

            for image in images:
                annotations = image['annotations']
                annotations = self._filter_by_categories(categories, annotations)
                annotations = self._filter_by_bbox_size(folder['bbox_limits'], (image['width'], image['height']),
                                                        annotations)
                annotations = self._filter_by_annotations_count(max_bbox_per_image, annotations)
                image['annotations'] = annotations
                final_images.append(image)
        if shuffle:
            random.shuffle(final_images)
        return final_images

    # Internals #
    def _load_all_folders(self):
        self._search_folders_recursively()

        self._train_index, train_folders = [], {}
        for folder in tqdm(self._train_folders, desc='Loading training folders', leave=False, miniters=1):
            images_index = self._load_folder(folder['path'])
            if images_index is not None:
                self._train_index.extend(images_index)
                train_folders[folder['path']] = {'images': images_index, **folder}
        self._train_folders = train_folders

        self._validation_index, validation_folders = [], {}
        for folder in tqdm(self._validation_folders, desc='Loading validation folders', leave=False, miniters=1):
            images_index = self._load_folder(folder['path'])
            if images_index is not None:
                self._validation_index.extend(images_index)
                validation_folders[folder['path']] = {'images': images_index, **folder}

        self._validation_folders = validation_folders

    def _search_folders_recursively(self):
        annotation_filenames = list(map(itemgetter(0), TrassirRectShapesAnnotations.ANNOTATIONS_FILENAMES))
        train_folders = []
        for folder in self._train_folders:
            if not folder['recursive']:
                train_folders.append(folder)
            else:
                folder_ops = deepcopy(folder)
                for cwd, folders, files in os.walk(folder['path']):
                    if any([ann_filename in files for ann_filename in annotation_filenames]):
                        folder_ops['path'] = cwd
                        train_folders.append(deepcopy(folder_ops))
        self._train_folders = train_folders

        validation_folders = []
        for folder in self._validation_folders:
            if not folder['recursive']:
                validation_folders.append(folder)
            else:
                folder_ops = deepcopy(folder)
                for cwd, folders, files in os.walk(folder['path']):
                    if any([ann_filename in files for ann_filename in annotation_filenames]):
                        folder_ops['path'] = cwd
                        validation_folders.append(deepcopy(folder_ops))
        self._validation_folders = validation_folders

    def _retrieve_categories(self):
        self._categories = []
        existing_categories_names = set()
        categories_id_by_name = {}

        common_index = self._train_index + self._validation_index
        for image in common_index:
            for ann in image['annotations']:
                if ann['category_name'] in existing_categories_names:
                    ann['category_id'] = categories_id_by_name[ann['category_name']]
                else:
                    next_id = max(list(categories_id_by_name.values())) + 1 if len(categories_id_by_name) > 0 else 0
                    existing_categories_names.add(ann['category_name'])
                    categories_id_by_name[ann['category_name']] = next_id
                    self._categories.append({'id': next_id, 'name': ann['category_name']})
                    ann['category_id'] = categories_id_by_name[ann['category_name']]

    def _retrieve_verifiers(self):
        self._verifiers = {}

        common_index = self._train_index + self._validation_index
        for image in common_index:
            for verifier_name, verified in image['verified'].items():
                if verifier_name in self._verifiers and verified:
                    self._verifiers[verifier_name].append(image['id'])
                elif verifier_name not in self._verifiers:
                    if not verified:
                        self._verifiers[verifier_name] = []
                    else:
                        self._verifiers[verifier_name] = [image['id']]

    def _build_extra_indexes(self):
        common_index = self._train_index + self._validation_index
        self._images_by_id_index = {image['id']: image for image in common_index}
        self._categories_by_id_index = {cat['id']: cat for cat in self._categories}
        self._categories_by_name_index = {cat['name']: cat for cat in self._categories}

    @staticmethod
    def _annotation_converter(annotation):
        annotation = deepcopy(annotation)
        assert annotation['shape_type'] == TrassirRectShapesAnnotations.ANNOTATION_TYPE
        del annotation['shape_type']
        del annotation['time_code']
        annotation['bbox'] = np.array(annotation['bbox'], dtype=np.float32)
        return annotation

    def _load_folder(self, path):
        if not os.path.exists(path):
            logging.error('Given path with annotations is not exist, skipping it: {}'.format(path))
            return None

        if all([not os.path.isfile(os.path.join(path, ann_filename))
                for ann_filename, _ in TrassirRectShapesAnnotations.ANNOTATIONS_FILENAMES]):
            logging.error('There is no annotation file in given path: {}'.format(path))
            return None

        annotations = None
        for ann_filename, module in TrassirRectShapesAnnotations.ANNOTATIONS_FILENAMES:
            if os.path.isfile(os.path.join(path, ann_filename)):
                with open(os.path.join(path, ann_filename), 'rb') as f:
                    annotations = module.load(f)
                break
        assert annotations is not None

        if not all(key in annotations for key in ['info', 'images', 'annotations', 'categories']):
            logging.error('Annotation file structure is invalid: {}'.format(path))
            return None

        if annotations['info']['version'] != TrassirRectShapesAnnotations.ANNOTATION_VERSION:
            logging.error('Annotation versions mismatch for {}: {} is required, {} is given'.format(
                path, TrassirRectShapesAnnotations.ANNOTATION_VERSION, annotations['info']['version']))
            return None

        images_by_id = {image_info['id']: {**image_info,
                                           'annotations': [],
                                           'file_name': os.path.join(path, image_info['file_name'])}
                        for image_info in annotations['images']}
        categories_by_id = {cat_info['id']: cat_info for cat_info in annotations['categories']}

        for ann in filter(lambda x: x['shape_type'] == TrassirRectShapesAnnotations.ANNOTATION_TYPE, annotations['annotations']):
            ann = self._annotation_converter(ann)
            ann['category_name'] = categories_by_id[ann['category_id']]['name']
            del ann['category_id']
            images_by_id[ann['image_id']]['annotations'].append(ann)

        for image in images_by_id.values():
            image['id'] = self._common_image_id
            self._common_image_id += 1

        return list(images_by_id.values())

    # Images filters #
    @staticmethod
    def _filter_by_image_availability(images):
        return [image_info
                for image_info in images
                if os.path.isfile(image_info['file_name'])]

    # If verifiers is None then we search for any verify signature
    def _filter_by_verifiers(self, verifiers, images):
        filtered_images_ids = set()
        if verifiers is None:
            for verifier in self._verifiers.keys():
                filtered_images_ids.update(self._verifiers[verifier])
        elif type(verifiers) is str:
            verifiers = [verifiers]
            for verifier in verifiers:
                if verifier in self._verifiers:
                    filtered_images_ids.update(self._verifiers[verifier])
        elif type(verifiers) is list:
            for verifier in verifiers:
                if verifier in self._verifiers:
                    filtered_images_ids.update(self._verifiers[verifier])
        else:
            assert False, "verifiers has invalid format in _filter_by_verifiers"

        return [
            image
            for image in images
            if image['id'] in filtered_images_ids
        ]

    # Annotations filters #
    # Categories - list of categories names
    @staticmethod
    def _filter_by_categories(categories, annotations):
        return [ann for ann in annotations if ann['category_name'] in categories]

    # bbox_size is tuple with all border min and max sizes, sizes is pixel count for given image_size
    @staticmethod
    def _filter_by_bbox_size(bbox_size, image_size, annotations):
        annotations = deepcopy(annotations)
        bbox_size = list(map(lambda x: np.iinfo(np.int32).max if x == 'inf' else x, bbox_size))
        min_w, min_h, max_w, max_h = bbox_size
        w, h = image_size

        bboxes_by_ids = [
            (ann['id'],
             [int(round(ann['bbox'][0][0] * w)), int(round(ann['bbox'][0][1] * h)),
              int(round(ann['bbox'][1][0] * w)), int(round(ann['bbox'][1][1] * h))])
            for ann in annotations
        ]

        filtered_bboxes_ids = set([
            i
            for i, bbox in bboxes_by_ids
            if (min_w <= bbox[2] - bbox[0] <= max_w) and
               (min_h <= bbox[3] - bbox[1] <= max_h)
            ])

        return [ann for ann in annotations if ann['id'] in filtered_bboxes_ids]

    # We will try to filter smallest boxes if there is too many annotations
    @staticmethod
    def _filter_by_annotations_count(max_count, annotations):
        if len(annotations) < max_count:
            return annotations

        annotations_with_areas = [
            (ann, (ann['bbox'][1][0] - ann['bbox'][0][0]) * (ann['bbox'][1][1] - ann['bbox'][0][1]))
            for ann in annotations
        ]

        annotations_with_areas = sorted(annotations_with_areas, reverse=True, key=lambda x: x[1])

        return [ann for ann, _ in annotations_with_areas[:max_count]]
