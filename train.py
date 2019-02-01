"""
Train the MobileNet V2 model
"""
import os
import sys
import argparse
import pandas as pd

from mobilenet_v2 import MobileNetv2

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model

import tensorflow as tf

from sklearn.metrics import classification_report

import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--classes",
        help="The number of classes of dataset.")
    # Optional arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=300,
        help="The number of train iterations.")
    parser.add_argument(
        "--weights",
        default=False,
        help="Fine tune with other weights.")
    parser.add_argument(
        "--tclasses",
        default=0,
        help="The number of classes of pre-trained model.")
    parser.add_argument(
        "--tflite",
        "-tl",
        action="store_true",
        help="The name of file to save the TFLite model")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs), int(args.classes), int(args.size), args.weights, int(args.tclasses), args.tflite)


def generate(batch, size):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    ptrain = 'data/train'
    pval = 'data/validation'

    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2


def fine_tune(num_classes, weights, model):
    """Re-build model with current num_classes.

    # Arguments
        num_classes, Integer, The number of classes of dataset.
        tune, String, The pre_trained model weights.
        model, Model, The model structure.
    """
    model.load_weights(weights)

    x = model.get_layer('Dropout').output
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    model = Model(inputs=model.input, outputs=output)

    return model

def keep_training(weights, model):
    model.load_weights(weights)
    return model

def create_callbacks():
    """
    # Arguments
        None
    """

    callbacks = [
        EarlyStopping(monitor='val_acc',
            patience=30,
            verbose=0,
            mode='auto',
            restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", 
            factor=0.5, 
            patience=10, 
            verbose=0,
            mode='auto',
            min_delta=0.00001,
            cooldown=0,
            min_lr=0)
    ]

    return callbacks

def generate_report(model, generator, batch, count):
    y_pred = model.predict_generator(generator, steps= count//batch)
    
    list_ = []

    b = count//batch

    for i in range(b):
        aux = generator[i]
        aux2 = 0
        for j in aux:
            aux2 += 1
            if aux2 % 2 == 0:
                for k in j:
                    list_.append(k.tolist())
    labels = [ i[0] for i in sorted(generator.class_indices.items(), key=lambda x: x[1])]
    print(classification_report(
        np.argmax(list_, axis=1),
        np.argmax(y_pred, axis=1),
        target_names = labels
        ))

def train(batch, epochs, num_classes, size, weights, tclasses, tflite):
    """Train the model.

    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
        tclasses, Integer, The number of classes of pre-trained model.
        tflite, Boolean, Convert the final model to a tflite model
    """

    train_generator, validation_generator, count1, count2 = generate(batch, size)

    if weights:
        if tclasses:
            print("fine tunning")
            model = MobileNetv2((size, size, 3), tclasses)
            model = fine_tune(num_classes, weights, model)
        else:
            print("Loading Weights")
            model = MobileNetv2((size, size, 3), num_classes)
            model = keep_training(weights, model)

    else:
        model = MobileNetv2((size, size, 3), num_classes)

    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=epochs,
        callbacks=create_callbacks())
    
    generate_report(model, validation_generator, batch, count2)

    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/hist.csv', encoding='utf-8', index=False)
    print("Saving weights")
    model.save_weights('model/weights.h5')
     
    model_name = "mobile_model.h5"

    if tflite:
        print("Saving model")
        model.save(model_name)
        print("Converting model")
        convert_to_lite(model_name)


def convert_to_lite(model, tflite_name="converted_model"):
    """
        Convert a saved model to tf lite format.

        # Arguments
            model: String, path to .h5 file model
    """
    tflite_name += ".tflite"
    converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model)
    tflite_model = converter.convert()
    open(tflite_name, "wb").write(tflite_model)

if __name__ == '__main__':
    main(sys.argv)
