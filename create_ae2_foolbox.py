#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import copy
import configparser
import numpy as np
import foolbox
import keras
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing import image
from foolbox.criteria import TargetClassProbability
from util import Utilty

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


# Create Adversarial examples.
class Adversarial:
    def __init__(self, utility):
        # Read config.ini.
        self.utility = utility
        config = configparser.ConfigParser()
        self.file_name = os.path.basename(__file__)
        self.full_path = os.path.dirname(os.path.abspath(__file__))
        config.read(os.path.join(self.full_path, 'config.ini'))

        try:
            # Define test dataset path.
            self.dataset_dir = os.path.join(self.full_path, config['Recognition']['dataset_path'])
            self.test_dir = os.path.join(self.dataset_dir, config['Recognition']['test_path'])
            if os.path.exists(self.test_dir) is False:
                os.mkdir(self.test_dir)

            # Define class list.nb_train_samples
            self.classes = os.listdir(self.test_dir)
            self.nb_classes = len(self.classes)

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

            # Define haarcascade path.
            haarcascade_dir = os.path.join(self.full_path, config['Preparation']['haarcascade_path'])
            self.haarcascade_path = os.path.join(haarcascade_dir, 'haarcascade_frontalface_default.xml')

            # Adversarial examples path.
            self.adversarial_path = os.path.join(self.full_path, config['Adversarial']['adversarial_path'])
            if os.path.exists(self.adversarial_path) is False:
                os.mkdir(self.adversarial_path)
            self.origin_image_path = os.path.join(self.full_path, config['Adversarial']['origin_adversarial'])
            if os.path.exists(self.origin_image_path) is False:
                os.mkdir(self.origin_image_path)
        except Exception as e:
            self.utility.print_message(FAIL, 'Reading config.ini is failure : {}'.format(e))
            sys.exit(1)

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


# main
if __name__ == "__main__":
    utility = Utilty()
    adv = Adversarial(utility)

    # Load existing Keras model.
    keras.backend.set_learning_phase(0)
    keras_model = adv.prepare_test()

    # Exchange Keras model to foolbox model.
    fool_model = foolbox.models.KerasModel(keras_model, bounds=(0, 255))

    # Generate adversarial examples.
    target_list = os.listdir(adv.origin_image_path)
    for idx1, target_origin_image in enumerate(target_list):
        # Extract label of target image.
        label = int(target_origin_image.split('.')[0])

        # Load target image.
        utility.print_message(OK, '{}/{} Load original image: {} = {}'.format(label + 1,
                                                                              len(target_list),
                                                                              target_origin_image,
                                                                              adv.classes[label]))
        origin_image = image.img_to_array(image.load_img(os.path.join(adv.origin_image_path, target_origin_image),
                                                         target_size=(adv.pixel_size, adv.pixel_size)))

        # Specify the target label.
        for idx2, target_class in enumerate(reversed(range(adv.nb_classes))):
            # Indicate target label.
            criterion = TargetClassProbability(label, p=0.9)
            attack = foolbox.attacks.LBFGSAttack(model=fool_model, criterion=criterion)
            utility.print_message(OK, 'Run the attack: target={}.{}'.format(label, adv.classes[label]))

            # Run the attack.
            adversarial = attack(origin_image, label=label, unpack=False)

            # Save and show adversarial examples.
            utility.print_message(OK, 'Show the images.')
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.title('Original')
            plt.imshow(origin_image / 255)

            plt.subplot(1, 3, 2)
            plt.title('Difference')
            difference = adversarial.image - origin_image
            plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)

            plt.subplot(1, 3, 3)
            plt.title('Adversarial')
            plt.imshow(adversarial.image / 255)

            # Prediction.
            # Original model.
            utility.print_message(NOTE, 'Prediction of original model.')
            pred = keras_model.predict(np.expand_dims(origin_image / 255, axis=0))[0]
            top_index = pred.argsort()[-1:][::-1]
            orig_pred_label = adv.classes[top_index[0]]
            orig_pred_prob = pred[top_index[0]]
            pred = keras_model.predict(np.expand_dims(adversarial.image / 255, axis=0))[0]
            top_index = pred.argsort()[-1:][::-1]
            adv_pred_label = adv.classes[top_index[0]]
            adv_pred_prob = pred[top_index[0]]
            msg = 'Original: {}/{:.1f}%, Adversarial: {}/{:.1f}%'.format(orig_pred_label, orig_pred_prob * 100,
                                                                         adv_pred_label, adv_pred_prob * 100)
            utility.print_message(OK, msg)

            # Foolbox model.
            utility.print_message(NOTE, 'Prediction of foolbox model.')
            pred_index = int(np.argmax(fool_model.predictions(origin_image)))
            orig_pred_label = adv.classes[pred_index]
            orig_pred_prob = foolbox.utils.softmax(fool_model.predictions(origin_image))[pred_index]
            pred_index = int(np.argmax(fool_model.predictions(adversarial.image)))
            adv_pred_label = adv.classes[pred_index]
            adv_pred_prob = foolbox.utils.softmax(fool_model.predictions(adversarial.image))[pred_index]
            msg = 'Original: {}/{:.1f}%, Adversarial: {}/{:.1f}%'.format(orig_pred_label, orig_pred_prob * 100,
                                                                         adv_pred_label, adv_pred_prob * 100)
            utility.print_message(OK, msg)

            # Save.
            file_name = 'Adversarial_{}.{}---{}.{}.jpg'.format(idx1, orig_pred_label, idx2, adv_pred_label)
            plt.imsave(os.path.join(adv.adversarial_path, file_name), adversarial.image / 255)
            file_name = 'Compare_{}.{}---{}.{}.jpg'.format(idx1, orig_pred_label, idx2, adv_pred_label)
            plt.savefig(os.path.join(adv.adversarial_path, file_name))
