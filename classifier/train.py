import importlib
import os

from keras.callbacks import LearningRateScheduler
from keras.engine.saving import load_model
from keras.layers import Input
from keras.models import Model

from callback import CustomModelCheckpoint
from classifier.generator import ClassifierBatchGenerator
from classifier.model import create_classifier_model, focal_loss
from preprocessing import TrassirRectShapesAnnotations
from yolo import RegressionLayer


def create_callbacks(classifier_model):
    checkpoint_weights = CustomModelCheckpoint(
        model_to_save=classifier_model,
        filepath='classifier.h5',
        monitor='accuracy',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        period=1)

    reduce_lrt = LearningRateScheduler(lambda epoch: 0.0001 if epoch < 100 else 0.0001)

    return [checkpoint_weights, reduce_lrt]


if __name__ == '__main__':
    config = importlib.import_module('config').config
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['cuda_devices']

    annotations = TrassirRectShapesAnnotations(
        config['datasets']['train'],
        config['datasets']['val'],
        categories=config['categories'],
        skip_categories=[])
    annotations.load()
    annotations.print_statistics()

    train = annotations.get_train_instances(verifiers=config['verifiers'],
                                            max_bbox_per_image=config['train']['max_roi_count'])
    validation = annotations.get_validation_instances(verifiers=config['verifiers'],
                                                      max_bbox_per_image=config['train']['max_roi_count'])

    train_generator = ClassifierBatchGenerator(train,
                                               config['categories'],
                                               (config['train']['net_size'], config['train']['net_size']),
                                               config['train']['max_roi_count'],
                                               config['train']['batch_size'])
    val_generator = ClassifierBatchGenerator(validation,
                                             config['categories'],
                                             (config['train']['net_size'], config['train']['net_size']),
                                             config['train']['max_roi_count'],
                                             config['train']['batch_size'])

    infer_model = load_model(config['infer_model_path'], custom_objects={'RegressionLayer': RegressionLayer})
    for layer in infer_model.layers:
        layer.trainable = False
    C2 = infer_model.get_layer('leaky_10').output
    C3 = infer_model.get_layer('leaky_35').output
    C4 = infer_model.get_layer('leaky_60').output
    C5 = infer_model.get_layer('leaky_78').output

    classifier_model = create_classifier_model(image_size=(config['train']['net_size'], config['train']['net_size'], 3),
                                               classes_count=len(config['categories']))
    if config['train']['classifier_weights_path'] is not None:
        classifier_model.load_weights(config['train']['classifier_weights_path'])
        print('Loaded {} weights'.format(config['train']['classifier_weights_path']))
    classifier_model.summary()

    m_roi_input = Input(shape=(config['train']['max_roi_count'], 4), name='input_2')
    x = classifier_model([C2, C3, C4, C5, m_roi_input])

    common_model = Model(inputs=[infer_model.inputs[0], m_roi_input], outputs=x)
    common_model.summary()
    common_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    callbacks = create_callbacks(classifier_model)

    common_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=config['train']['epochs'],
        verbose=1,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        workers=4,
        max_queue_size=16
    )
