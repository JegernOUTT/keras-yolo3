import importlib
import os

from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

from callback import CustomModelCheckpoint
from classifier.generator import ClassifierBatchGenerator
from classifier.model import create_classifier_model, focal_loss
from preprocessing import TrassirRectShapesAnnotations


def create_callbacks(classifier_model):
    checkpoint_weights = CustomModelCheckpoint(
        model_to_save=classifier_model,
        filepath='classifier.h5',
        monitor='accuracy',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        period=1)


    return [checkpoint_weights]


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

    train_generator = ClassifierBatchGenerator(instances=train,
                                               classes=config['categories'],
                                               net_image_size=(config['train']['net_size'], config['train']['net_size']),
                                               is_train=True,
                                               batch_size=config['train']['batch_size'])
    val_generator = ClassifierBatchGenerator(instances=validation,
                                             classes=config['categories'],
                                             net_image_size=(config['train']['net_size'], config['train']['net_size']),
                                             is_train=False,
                                             batch_size=config['train']['batch_size'])

    classifier_model = create_classifier_model(image_shape=(config['train']['net_size'], config['train']['net_size'], 3),
                                               classes_count=len(config['categories']))
    if config['train']['classifier_weights_path'] is not None:
        classifier_model.load_weights(config['train']['classifier_weights_path'])
        print('Loaded {} weights'.format(config['train']['classifier_weights_path']))
    classifier_model.summary()

    classifier_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=config['train']['epochs'],
        verbose=1,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=create_callbacks(classifier_model),
        workers=4,
        max_queue_size=16
    )
