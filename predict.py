#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import configparser
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


# Face prediction.
class Recognition:
    def __init__(self, utility):
        # Read config.ini.
        self.utility = utility
        config = configparser.ConfigParser()
        self.file_name = os.path.basename(__file__)
        self.full_path = os.path.dirname(os.path.abspath(__file__))
        config.read(os.path.join(self.full_path, 'config.ini'))

        try:
            # Define train dataset path.
            self.dataset_dir = os.path.join(self.full_path, config['Recognition']['dataset_path'])
            self.train_dir = os.path.join(self.dataset_dir, config['Recognition']['train_path'])
            if os.path.exists(self.train_dir) is False:
                os.mkdir(self.train_dir)

            # Define test dataset path.
            self.test_dir = os.path.join(self.dataset_dir, config['Recognition']['test_path'])
            if os.path.exists(self.test_dir) is False:
                os.mkdir(self.test_dir)

            # Define class list.
            self.classes = os.listdir(self.test_dir)
            self.nb_classes = len(self.classes)

            # Parameters.
            self.batch_size = int(config['Recognition']['batch_size'])
            self.nb_classes = len(self.classes)

            # Training information.
            self.nb_train_samples, self.nb_val_samples = self.count_data_num()
            self.nb_epoch = int(config['Recognition']['epoch_num'])

            # Define model path.
            self.model_dir = os.path.join(self.dataset_dir, config['Recognition']['model_path'])
            if os.path.exists(self.model_dir) is False:
                os.mkdir(self.model_dir)
            self.model_name = os.path.join(self.model_dir, config['Recognition']['model_name'])

            # Pixel size.
            self.pixel_size = int(config['Preparation']['pic_size'])

            # Image information.
            self.img_rows, self.img_cols = self.pixel_size, self.pixel_size
            self.channels = 3
        except Exception as e:
            self.utility.print_message(FAIL, 'Reading config.ini is failure : {}'.format(e))
            sys.exit(1)

    # Count train and test data.
    def count_data_num(self):
        # Count train data.
        train_count = 0
        for root, dirs, files in os.walk(self.train_dir):
            train_count += len(files)

        # Count test data.
        test_count = 0
        for root, dirs, files in os.walk(self.test_dir):
            test_count += len(files)

        return train_count, test_count

    # Build VGG16 model.
    def build_vgg16(self):
        # VGG16
        self.utility.print_message(OK, 'Build VGG16 model.')
        input_tensor = Input(shape=(self.img_rows, self.img_cols, self.channels))
        vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        return vgg16

    # Build FC.
    def build_fc(self, vgg16):
        # FC
        self.utility.print_message(OK, 'Build FC model.')
        fc = Sequential()
        fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
        fc.add(Dense(256, activation='relu'))
        fc.add(Dropout(0.5))
        fc.add(Dense(self.nb_classes, activation='softmax'))
        return fc

    # Connect VGG16 and FC.
    def connect_vgg16_fc(self, vgg16, fc):
        # Connect VGG16 and FC.
        self.utility.print_message(OK, 'Connect VGG16 and FC.')
        model = Model(input=vgg16.input, output=fc(vgg16.output))
        return model

    # Load trained model.
    def load_model(self, model):
        # Load model (trained).
        self.utility.print_message(OK, 'Load trained model: {}'.format(self.model_name))
        model.load_weights(self.model_name)
        return model

    # Execute training.
    def execute_train(self):
        self.utility.print_message(NOTE, 'Start training model.')

        # Build VGG16.
        vgg16 = self.build_vgg16()

        # Build FC.
        fc = self.build_fc(vgg16)

        # Connect VGG16 and FC.
        model = self.connect_vgg16_fc(vgg16, fc)

        # Freeze before last layer.
        for layer in model.layers[:15]:
            layer.trainable = False

        # Use Loss=categorical_crossentropy.
        self.utility.print_message(OK, 'Compile model.')
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_rows, self.img_cols),
            color_mode='rgb',
            classes=self.classes,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True)

        validation_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_rows, self.img_cols),
            color_mode='rgb',
            classes=self.classes,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True)

        # Fine-tuning.
        self.utility.print_message(OK, 'Execute fine-tuning.')
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=self.nb_train_samples,
            nb_epoch=self.nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=self.nb_val_samples)

        # Save model.
        self.utility.print_message(NOTE, 'Save model to {}'.format(self.model_name))
        model.save_weights(self.model_name)
        self.utility.print_message(NOTE, 'Finish training model.')

    # Prepare test.
    def prepare_test(self):
        # Build VGG16.
        vgg16 = self.build_vgg16()

        # Build FC.
        fc = self.build_fc(vgg16)

        # Connect VGG16 and FC.
        model = self.connect_vgg16_fc(vgg16, fc)

        # Load model.
        model = self.load_model(model)

        # Use Loss=categorical_crossentropy.
        self.utility.print_message(OK, 'Compile model.')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # Execute prediction.
    def execute_test(self, model, in_image):
        # Transform image to 4 dimension tensor.
        img = image.load_img(in_image, target_size=(self.img_rows, self.img_cols))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Get result of prediction.
        pred = model.predict(x)[0]
        top = 1
        top_indices = pred.argsort()[-top:][::-1]
        results = [(self.classes[i], pred[i]) for i in top_indices]

        # Predicted label and probability.
        return results
