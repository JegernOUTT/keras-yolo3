import importlib
import os

import keras.backend as K
from keras.engine.saving import load_model
from keras.layers import Input
from keras.models import Model

from classifier.generator import ClassifierBatchGenerator
from classifier.model import create_classifier_model
from preprocessing import TrassirRectShapesAnnotations
from yolo import RegressionLayer

if __name__ == '__main__':
    K.set_learning_phase(0)

    config = importlib.import_module('config').config
    os.environ['CUDA_VISIBLE_DEVICES'] = config['eval']['cuda_devices']

    annotations = TrassirRectShapesAnnotations(
        [],
        config['datasets']['val'],
        categories=config['categories'],
        skip_categories=[])
    annotations.load()
    annotations.print_statistics()
    validation = annotations.get_validation_instances(verifiers=config['verifiers'],
                                                      max_bbox_per_image=config['eval']['max_roi_count'])
    val_generator = ClassifierBatchGenerator(validation,
                                             config['categories'],
                                             (config['eval']['net_size'], config['eval']['net_size']),
                                             config['eval']['max_roi_count'],
                                             1)

    infer_model = load_model(config['infer_model_path'], custom_objects={'RegressionLayer': RegressionLayer})
    C2 = infer_model.get_layer('leaky_10').output
    C3 = infer_model.get_layer('leaky_35').output
    C4 = infer_model.get_layer('leaky_60').output
    C5 = infer_model.get_layer('leaky_78').output

    classifier_model = create_classifier_model(image_size=(config['eval']['net_size'], config['eval']['net_size'], 3),
                                               classes_count=len(config['categories']))
    assert os.path.exists(config['eval']['classifier_weights_path'])
    classifier_model.load_weights(config['eval']['classifier_weights_path'])
    classifier_model.summary()

    m_roi_input = Input(shape=(config['eval']['max_roi_count'], 4), name='input_2')
    x = classifier_model([C2, C3, C4, C5, m_roi_input])

    common_model = Model(inputs=[infer_model.inputs[0], m_roi_input], outputs=x)
    common_model.summary()
    common_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    metrics = common_model.evaluate_generator(val_generator)
    print('Loss: {:.4f}, Accuracy: {:.4f}'.format(*metrics))
