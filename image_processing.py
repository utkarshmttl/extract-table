import math

import cv2
import numpy as np
from constants import RED

# Threshold for length for a black area to be considered a line
LINE_LENGTH = 12

# Line drawn after table detection
LINE_THICKNESS = 3


def extract_red_from_image(gray_version, image):
    gray_version[(image == RED).all(axis=2)] = 0  # extract only red pixels of img

    gray_version = gray_version.astype('uint8')
    return gray_version


def detect_horizontal_and_vertical_lines(grayscale_image, image):
    lsd = cv2.createLineSegmentDetector(0)
    detected_lines = lsd.detect(grayscale_image)

    for detected_line in detected_lines[0]:
        x0 = round(detected_line[0][0])
        y0 = round(detected_line[0][1])
        x1 = round(detected_line[0][2])
        y1 = round(detected_line[0][3])

        # Use slope to determine if line is horizontal or vertical
        if abs(x1 - x0) < 0.1 or abs((y1 - y0) / (x1 - x0)) < 0.1:
            a = (x0 - x1) * (x0 - x1)
            b = (y0 - y1) * (y0 - y1)
            c = a + b
            length = math.sqrt(c)

            if length > LINE_LENGTH:
                cv2.line(image, (x0, y0), (x1, y1), RED, LINE_THICKNESS, cv2.LINE_AA)

    return image


def get_a_white_clone(image):
    return np.ones(image[:, :, 0].shape) * 255  # pure white
