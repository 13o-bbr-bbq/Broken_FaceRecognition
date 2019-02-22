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
    if len(sys.argv) < 2:
        print('Target image not found.')
        sys.exit(1)

    utility = Utilty()
    adv = Adversarial(utility)

    # Instantiate model
    keras.backend.set_learning_phase(0)
    kmodel = adv.prepare_test()
    preprocessing = (np.array([104, 116, 123]), 1)
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

    # Load target image.
    target_image = sys.argv[1]
    utility.print_message(OK, 'Load original image: {}'.format(target_image))
    predict_image = image.load_img(target_image, target_size=(adv.pixel_size, adv.pixel_size))
    origin_image = image.img_to_array(predict_image)

    # Specify the target label.
    target_class = 4
    criterion = TargetClassProbability(target_class, p=0.95)

    # Run the attack.
    utility.print_message(OK, 'Run the attack: target={}.{}'.format(target_class, adv.classes[target_class]))
    attack = foolbox.attacks.LBFGSAttack(model=fmodel, criterion=criterion)
    adversarial = attack(origin_image, label=target_class)

    # Prediction of default model.
    copy_image = copy.deepcopy(origin_image)
    copy_image = np.expand_dims(copy_image, axis=0)
    copy_image = copy_image / 255.0
    pred = kmodel.predict(copy_image)[0]
    top_indices = pred.argsort()[-1:][::-1]
    results = [(adv.classes[i], pred[i]) for i in top_indices]
    pred_label = adv.classes[int(np.argmax(fmodel.predictions(adversarial)))]
    pred_prob = foolbox.utils.softmax(fmodel.predictions(adversarial))[4]
    utility.print_message(OK, 'Prediction result for adversarial: {}/{:.1f}%'.format(pred_label, pred_prob * 100))

    utility.print_message(OK, 'Show the images.')
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(origin_image / 255)

    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow(adversarial / 255)
    plt.imsave('adv_example.png', adversarial / 255)

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = adversarial - origin_image
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)

    plt.show()
