#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import configparser
import cv2
from docopt import docopt
from util import Utilty
from prepare import Preparation
from predict import Recognition

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.

# Define command option.
__doc__ = """{f}
usage:
    {f} [-g <label_name>] [-c] [-t] [-a]
    {f} -h | --help
options:
    -g   Optional : Gather your face images.
    -c   Optional : Create dataset that train and test.
    -t   Optional : Train face recognition model.
    -a   Optional : Adversarial Examples test.
    -h --help     Show this help message and exit.
""".format(f=__file__)


# Parse command arguments.
def command_parse(utility):
    utility.write_log(20, '[In] Parse command options [{}].'.format(os.path.basename(__file__)))

    args = docopt(__doc__)
    opt_gather = args['-g']
    opt_label_name = args['<label_name>']
    opt_create = args['-c']
    opt_train = args['-t']
    opt_adversarial = args['-a']

    utility.write_log(20, '[Out] Parse command options [{}].'.format(os.path.basename(__file__)))
    return opt_gather, opt_label_name, opt_create, opt_train, opt_adversarial


if __name__ == '__main__':
    file_name = os.path.basename(__file__)
    full_path = os.path.dirname(os.path.abspath(__file__))

    utility = Utilty()
    utility.write_log(20, '[In] Broken Face recognition [{}].'.format(file_name))

    # Get command arguments.
    opt_gather, opt_label_name, opt_create, opt_train, opt_adversarial = command_parse(utility)

    # Read config.ini.
    config = configparser.ConfigParser()
    config.read(os.path.join(full_path, 'config.ini'))

    # Getting sample number.
    max_retry = int(config['Authorization']['max_retry'])
    threshold = float(config['Authorization']['threshold'])

    # Default label name.
    default_label_name = config['Preparation']['default_label_name']

    # Adversarial Examples path.
    adversarial_path = os.path.join(full_path, config['Adversarial']['adversarial_path'])
    adversarial_result = os.path.join(full_path, config['Adversarial']['result_path'])

    # Create instance.
    preparation = Preparation(utility)
    recognition = Recognition(utility)

    if opt_train:
        # Train model.
        recognition.execute_train()
    elif opt_gather:
        # Gather registration face images.
        if opt_label_name == '':
            opt_label_name = default_label_name

        capture = preparation.create_camera_instance()
        preparation.gather_your_face(capture, opt_label_name)
    elif opt_create:
        # Create dataset.
        preparation.create_dataset()
    elif opt_adversarial:
        # Load model.
        model = recognition.prepare_test()
        target_list = os.listdir(adversarial_path)
        for idx, adversarial_example in enumerate(target_list):
            # Load adversarial example.
            image = cv2.imread(os.path.join(adversarial_path, adversarial_example), cv2.IMREAD_UNCHANGED)

            # Execute detecting face.
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(preparation.haarcascade_path)
            faces = cascade.detectMultiScale(gray_image,
                                             scaleFactor=1.1,
                                             minNeighbors=2,
                                             minSize=(preparation.pixel_size, preparation.pixel_size))

            if len(faces) == 0:
                utility.print_message(WARNING, 'Face is not found.')
                continue

            for face_idx, face in enumerate(faces):
                # Extract face information.
                x, y, width, height = face
                predict_image = image[y:y + height, x:x + width]
                if predict_image.shape[0] < preparation.pixel_size:
                    continue
                predict_image = cv2.resize(predict_image, (preparation.pixel_size, preparation.pixel_size))

                # Save image.
                file_name = os.path.join(adversarial_result, 'tmp_face.jpg')
                cv2.imwrite(file_name, predict_image)

                # Predict face.
                results = recognition.execute_test(model, file_name)

                prob = results[0][1] * 100
                msg = '{}/{} {} ({:.1f}%).'.format(idx + 1, len(target_list), results[0][0], prob)
                utility.print_message(OK, msg)

                # Draw frame to face.
                cv2.rectangle(image,
                              (x, y),
                              (x + width, y + height),
                              (255, 255, 255),
                              thickness=2)

                # Get current date.
                date = utility.get_current_date('%Y%m%d%H%M%S%f')[:-3]
                print_date = utility.transform_date_string(utility.transform_date_object(date[:-3], '%Y%m%d%H%M%S'))

                # Display raw frame data.
                cv2.putText(image, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, print_date, (10, 730), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Adversarial Example test.', image)

                file_name = os.path.join(adversarial_result, 'tmp_face' + str(idx) + '-' + str(face_idx) + '.jpg')
                cv2.imwrite(file_name, image)

            # Waiting for getting key input.
            k = cv2.waitKey(500)
            if k == 27:
                break
    else:
        # Execute face recognition.
        model = recognition.prepare_test()
        capture = preparation.create_camera_instance()
        for idx in range(max_retry):
            # Read 1 frame from VideoCapture.
            ret, image = capture.read()

            # Execute detecting face.
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(preparation.haarcascade_path)
            faces = cascade.detectMultiScale(gray_image,
                                             scaleFactor=1.1,
                                             minNeighbors=2,
                                             minSize=(preparation.pixel_size, preparation.pixel_size))

            if len(faces) == 0:
                utility.print_message(WARNING, 'Face is not found.')
                continue

            for face in faces:
                # Extract face information.
                x, y, width, height = face
                predict_image = image[y:y + height, x:x + width]
                if predict_image.shape[0] < preparation.pixel_size:
                    continue
                predict_image = cv2.resize(predict_image, (preparation.pixel_size, preparation.pixel_size))

                # Save image.
                file_name = os.path.join(preparation.dataset_path, 'tmp_face.jpg')
                cv2.imwrite(file_name, predict_image)

                # Predict face.
                results = recognition.execute_test(model, file_name)

                judge = 'Reject'
                prob = results[0][1] * 100
                if prob > threshold:
                    judge = 'Unlock'
                msg = '{} ({:.1f}%). res="{}"'.format(results[0][0], prob, judge)
                utility.print_message(OK, msg)

                # Draw frame to face.
                cv2.rectangle(image,
                              (x, y),
                              (x + width, y + height),
                              (255, 255, 255),
                              thickness=2)

                # Get current date.
                date = utility.get_current_date('%Y%m%d%H%M%S%f')[:-3]
                print_date = utility.transform_date_string(utility.transform_date_object(date[:-3], '%Y%m%d%H%M%S'))

                # Display raw frame data.
                cv2.putText(image, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, print_date, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Face Authorization', image)

                file_name = os.path.join(preparation.dataset_path, 'tmp_face' + str(idx) + '_.jpg')
                cv2.imwrite(file_name, image)

            # Waiting for getting key input.
            k = cv2.waitKey(500)
            if k == 27:
                break

        # Termination (release capture and close window).
        capture.release()
        cv2.destroyAllWindows()
