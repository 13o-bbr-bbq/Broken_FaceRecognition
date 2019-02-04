#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import configparser
import cv2
from docopt import docopt
from util import Utilty
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
    {f} [-t] [-g <label_name>] [-p]
    {f} -h | --help
options:
    -t   Optional : Train face recognition model.
    -g   Optional : Get registration data you want.
    -p   Optional : Prepare dataset.
    -h --help     Show this help message and exit.
""".format(f=__file__)


# Parse command arguments.
def command_parse(utility):
    utility.write_log(20, '[In] Parse command options [{}].'.format(os.path.basename(__file__)))

    args = docopt(__doc__)
    opt_train = args['-t']
    opt_get = args['-g']
    opt_label_name = args['<label_name>']
    opt_prepare = args['-p']

    utility.write_log(20, '[Out] Parse command options [{}].'.format(os.path.basename(__file__)))
    return opt_train, opt_get, opt_label_name, opt_prepare


if __name__ == '__main__':
    file_name = os.path.basename(__file__)
    full_path = os.path.dirname(os.path.abspath(__file__))

    utility = Utilty()
    utility.write_log(20, '[In] Broken Face recognition [{}].'.format(file_name))

    # Get command arguments.
    opt_train, opt_get, opt_label_name, opt_prepare = command_parse(utility)

    # Read config.ini.
    config = configparser.ConfigParser()
    config.read(os.path.join(full_path, 'config.ini'))

    # Define path.
    output_dir = os.path.join(full_path, config['Recognition']['dataset_path'])
    haarcascade_dir = os.path.join(full_path, 'haarcascade')

    # Getting sample number.
    samples = int(config['CreateSample']['samples'])

    # Getting sample number.
    max_retry = int(config['Authorization']['max_retry'])
    threshold = float(config['Authorization']['threshold'])

    # Create camera instance.
    capture = None
    try:
        capture = cv2.VideoCapture(0)
    except:
        utility.print_message(FAIL, 'Camera is not found.')
        sys.exit(1)

    if opt_train:
        # Train model.
        print('Train')
        sys.exit(0)
    elif opt_get:
        # Get registration samples for model training.
        for idx in range(samples):
            # Read 1 frame from VideoCapture.
            ret, image = capture.read()

            # Execute detecting face.
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(os.path.join(haarcascade_dir, 'haarcascade_frontalface_default.xml'))
            faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

            if len(faces) == 0:
                print('Face is not found.')
                continue

            for face in faces:
                # Extract face information.
                x, y, width, height = face
                register_image = image[y:y + height, x:x + width]
                if register_image.shape[0] < 64:
                    continue
                register_image = cv2.resize(register_image, (64, 64))

                # Display raw frame data.
                cv2.imshow('Register face image.', register_image)

                # Save image.
                file_name = os.path.join(output_dir, opt_label_name + '_' + str(idx) + '.jpg')
                cv2.imwrite(file_name, register_image)

            # Waiting for getting key input.
            k = cv2.waitKey(500)
            if k == 27:
                break
    elif opt_prepare:
        # Prepare dataset for training model.
        print('Prepare')
    else:
        # Execute face recognition.
        recognition = Recognition(utility)
        model = recognition.prepare_test()
        for idx in range(max_retry):
            # Read 1 frame from VideoCapture.
            ret, image = capture.read()

            # Execute detecting face.
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(os.path.join(haarcascade_dir, 'haarcascade_frontalface_default.xml'))
            faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

            if len(faces) == 0:
                print('Face is not found.')
                continue

            for face in faces:
                # Extract face information.
                x, y, width, height = face
                predict_image = image[y:y + height, x:x + width]
                if predict_image.shape[0] < 64:
                    continue
                predict_image = cv2.resize(predict_image, (64, 64))

                # Save image.
                file_name = os.path.join(output_dir, 'tmp_face.jpg')
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

                # Display raw frame data.
                cv2.putText(image, msg, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Face Authorization', image)

                #file_name = os.path.join(output_dir, 'tmp_face' + str(idx) + '_.jpg')
                #cv2.imwrite(file_name, image)

            # Waiting for getting key input.
            k = cv2.waitKey(500)
            if k == 27:
                break

    # Termination (release capture and close window).
    capture.release()
    cv2.destroyAllWindows()
