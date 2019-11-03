#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import configparser
import cv2

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


# Record movie using web camera.
class Record:
    def __init__(self, utility):
        # Read config.ini.
        self.utility = utility
        config = configparser.ConfigParser()
        self.file_name = os.path.basename(__file__)
        self.full_path = os.path.dirname(os.path.abspath(__file__))
        config.read(os.path.join(self.full_path, 'config.ini'))

        try:
            self.save_path = os.path.join(self.full_path, config['Record']['save_path'])
            self.fps = int(config['Record']['fps'])
            self.cap_wait_time = int(config['Record']['cap_wait_time'])
        except Exception as e:
            self.utility.print_message(FAIL, 'Reading config.ini is failure : {}'.format(e))
            sys.exit(1)

    # Setting of movie.
    def movie_setting(self, opt_rec_file_name):
        # Set web camera.
        capture = cv2.VideoCapture(0)

        # Setting of fps and frame size.
        window_size = (640, 480)

        # Setting of movie format.
        movie_format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(os.path.join(self.save_path, opt_rec_file_name), movie_format, self.fps, window_size)

        return capture, video

    # Start recording.
    def record(self, opt_rec_file_name):
        # Setting.
        capture, video = self.movie_setting(opt_rec_file_name)

        # Record.
        while capture.isOpened():
            # Capture one frame.
            ret, frame = capture.read()

            # Display.
            cv2.imshow('Now recording..', frame)

            # Write a frame to movie.
            video.write(frame)

            # Quit.
            # Waiting for getting key input.
            k = cv2.waitKey(self.cap_wait_time)
            if k == 27:
                break

        # Termination.
        capture.release()
        video.release()
        cv2.destroyAllWindows()
