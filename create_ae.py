#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import random
import copy
import cv2


# ランダム選択したピクセルに細工を加える
def random_adv(image, face, p):
    # Extract face information.
    x, y, width, height = face

    for _ in range(p):
        # 細工するピクセルをランダムに選択
        target_x = random.randint(x, width - 1)
        target_y = random.randint(y, height - 1)

        # ピクセル値の細工
        target_pixel = image[target_y, target_x]
        average = sum(target_pixel) / len(target_pixel)

        if average < 128:
            image[target_y, target_x] = [0, 0, 0]
        else:
            image[target_y, target_x] = [255, 255, 255]

    return image


if __name__ == "__main__":
    full_path = os.path.dirname(os.path.abspath(__file__))

    # Setting 細工するピクセル数(p)、1画像たりの細工試行最大回数(p_max_num)
    p = 1500
    p_max_num = 500

    if len(sys.argv) != 2:
        sys.exit(1)

    # Load original image.
    target_image = os.path.join(full_path, sys.argv[1])
    image = cv2.imread(target_image, cv2.IMREAD_UNCHANGED)

    # Execute face detection.
    haarcascade_path = os.path.join(os.path.join(full_path, 'haarcascade'), 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(haarcascade_path)
    faces = cascade.detectMultiScale(gray_image,
                                     scaleFactor=1.1,
                                     minNeighbors=2,
                                     minSize=(64, 64))

    if len(faces) == 0:
        print('Face is not found.')
    else:
        out_path = os.path.join(full_path, 'adversarial_examples')

        for face in faces:
            # 細工処理
            for idx, count in enumerate(range(p_max_num)):
                copy_image = copy.deepcopy(image)
                adv_image = random_adv(copy_image, face, p)

                # Save Adversarial Examples.
                save_path = os.path.join(out_path, 'adv_' + str(idx) + '_' + sys.argv[1])
                cv2.imwrite(save_path, adv_image)
                print('Save adversarial example: {}'.format(save_path))
