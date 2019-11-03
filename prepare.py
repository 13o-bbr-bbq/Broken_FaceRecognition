#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import shutil
import cv2
import glob
import configparser

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


# Preparation.
class Preparation:
    def __init__(self, utility):
        # Read config.ini.
        self.utility = utility
        config = configparser.ConfigParser()
        self.file_name = os.path.basename(__file__)
        self.full_path = os.path.dirname(os.path.abspath(__file__))
        config.read(os.path.join(self.full_path, 'config.ini'))

        try:
            # Camera instance number.
            self.camera_instance_num = int(config['Preparation']['camera_instance_num'])

            # Define origin data path.
            self.input_dir = os.path.join(self.full_path, config['Preparation']['origin_data_path'])
            if os.path.exists(self.input_dir) is False:
                self.utility.print_message(FAIL, 'Data is not found: {}'.format(self.input_dir))
                sys.exit(1)

            # Define dataset path.
            self.dataset_path = os.path.join(self.full_path, config['Recognition']['dataset_path'])
            if os.path.exists(self.dataset_path) is False:
                os.mkdir(self.dataset_path)

            # Define train and test dir.
            self.train_dir = os.path.join(self.dataset_path, config['Recognition']['train_path'])
            if os.path.exists(self.train_dir) is False:
                os.mkdir(self.train_dir)

            # Define test dataset path.
            self.test_dir = os.path.join(self.dataset_path, config['Recognition']['test_path'])
            if os.path.exists(self.test_dir) is False:
                os.mkdir(self.test_dir)

            # Define haarcascade path.
            haarcascade_dir = os.path.join(self.full_path, config['Preparation']['haarcascade_path'])
            self.haarcascade_path = os.path.join(haarcascade_dir, 'haarcascade_frontalface_default.xml')
            if os.path.exists(self.haarcascade_path) is False:
                self.utility.print_message(FAIL, 'Haarcascade file is not found: {}'.format(self.haarcascade_path))
                sys.exit(1)

            # Separation rate that train data and test data.
            self.separate_rate = float(config['Preparation']['separate_rate'])

            # Gathering sample number.
            self.sample_num = int(config['Preparation']['gather_samples'])

            # Pixel size.
            self.pixel_size = int(config['Preparation']['pic_size'])

            # Capture wait time (ms).
            self.cap_wait_time = int(config['Preparation']['cap_wait_time'])
        except Exception as e:
            self.utility.print_message(FAIL, 'Reading config.ini is failure : {}'.format(e))
            sys.exit(1)

    # Create camera instance.
    def create_camera_instance(self):
        # Create camera instance.
        capture = None
        try:
            capture = cv2.VideoCapture(self.camera_instance_num)
        except:
            self.utility.print_message(FAIL, 'Camera is not found.')
            sys.exit(1)
        return capture

    # Gather your face image.
    def gather_your_face(self, capture, label_name):
        self.utility.print_message(NOTE, 'Start gathering your face images.')

        # Gather registration your face image.
        save_path = os.path.join(self.input_dir, label_name)
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)
        for idx in range(self.sample_num):
            # Read 1 frame from VideoCapture.
            self.utility.print_message(OK, '{}/{} Capturing face image.'.format(idx + 1, self.sample_num))
            ret, image = capture.read()

            # Execute detecting face.
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(self.haarcascade_path)
            faces = cascade.detectMultiScale(gray_image,
                                             scaleFactor=1.1,
                                             minNeighbors=2,
                                             minSize=(self.pixel_size, self.pixel_size))

            if len(faces) == 0:
                self.utility.print_message(WARNING, 'Face is not found.')
                continue

            for face in faces:
                # Extract face information.
                x, y, width, height = face
                face_size = image[y:y + height, x:x + width]
                if face_size.shape[0] < self.pixel_size:
                    msg = 'This face is too small: {} pixel.'.format(str(face_size.shape[0]))
                    self.utility.print_message(WARNING, msg)
                    continue

                # Save image.
                file_name = os.path.join(save_path, label_name + '_' + str(idx) + '.jpg')
                cv2.imwrite(file_name, image)

                # Display raw frame data.
                cv2.rectangle(image,
                              (x, y),
                              (x + width, y + height),
                              (255, 255, 255),
                              thickness=2)

                # Display raw frame data.
                msg = 'Captured {}/{}.'.format(idx + 1, self.sample_num)
                cv2.putText(image, msg, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Captured your face', image)

            # Waiting for getting key input.
            k = cv2.waitKey(self.cap_wait_time)
            if k == 27:
                break

        # Termination (release capture and close window).
        capture.release()
        cv2.destroyAllWindows()
        self.utility.print_message(NOTE, 'Finish gathering your face images.')

    # Create dataset.
    def create_dataset(self):
        self.utility.print_message(NOTE, 'Start creating dataset.')

        # Execute face recognition in saved image.
        label_list = os.listdir(self.input_dir)
        for label in label_list:
            # Extract target image each label.
            target_dir = os.path.join(self.input_dir, label)
            in_image = glob.glob(os.path.join(target_dir, '*'))

            # Detect face in image.
            for idx, image in enumerate(in_image):
                # Read image to OpenCV.
                cv_image = cv2.imread(image)

                if cv_image.shape[0] < self.pixel_size:
                    msg = 'This face is too small: {} pixel.'.format(str(cv_image.shape[0]))
                    self.utility.print_message(WARNING, msg)
                    continue
                save_image = cv2.resize(cv_image, (self.pixel_size, self.pixel_size))

                # Save image.
                file_name = os.path.join(self.dataset_path, label + '_' + str(idx) + '.jpg')
                cv2.imwrite(file_name, save_image)

        # Separate images to train and test.
        for label in label_list:
            # Define train and test dir each label.
            train_label_dir = os.path.join(self.train_dir, label)
            if os.path.exists(train_label_dir) is False:
                os.mkdir(train_label_dir)
            test_label_dir = os.path.join(self.test_dir, label)
            if os.path.exists(test_label_dir) is False:
                os.mkdir(test_label_dir)

            # Get images of label.
            in_image = glob.glob(os.path.join(self.dataset_path, label + '*' + '.jpg'))
            for idx, image in enumerate(in_image):
                if idx < len(in_image) * self.separate_rate:
                    shutil.move(image, train_label_dir)
                else:
                    shutil.move(image, test_label_dir)

        self.utility.print_message(NOTE, 'Finish creating dataset.')
