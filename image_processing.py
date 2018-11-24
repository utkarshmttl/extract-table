import math

import cv2
import numpy as np

from constants import RED, LINE_LENGTH, MAX_LINE_LENGTH_FOR_GAP, SCALING_FACTOR

# Line drawn after table detection
LINE_THICKNESS = 3


def convert_to_grayscale(image_file):
    gray = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY)
    (_, gray) = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    return gray


def extract_lines_from_image(image):
    clone = get_a_white_clone(image)
    if len(image.shape) == 3:
        clone[(image == RED).all(axis=2)] = 0  # extract only red pixels of img
    else:
        clone[(image < 255)] = 0  # extract only red pixels of img

    gray_version = clone.astype('uint8')
    return gray_version


def scale_dimension(dim1, dim2, factor):
    base_length = dim2 - dim1
    ret1 = dim1 - (base_length * (factor-1) / 2)
    ret2 = dim2 + (base_length * (factor-1) / 2)
    return ret1, ret2


def scale(px1, px2, py1, py2, factor):
    x1, x2 = scale_dimension(px1, px2, factor)
    y1, y2 = scale_dimension(py1, py2, factor)
    return int(x1), int(y1), int(x2), int(y2)


def detect_horizontal_and_vertical_lines_with_color(grayscale_image, image):
    lsd = cv2.createLineSegmentDetector(0)
    detected_lines = lsd.detect(grayscale_image)

    for detected_line in detected_lines[0]:
        x0 = round(detected_line[0][0])
        y0 = round(detected_line[0][1])
        x1 = round(detected_line[0][2])
        y1 = round(detected_line[0][3])

        # Use slope to determine if line is horizontal or vertical
        # if abs(x1 - x0) < 0.2 or abs((y1 - y0) / (x1 - x0)) < 0.2:
        a = (x0 - x1) * (x0 - x1)
        b = (y0 - y1) * (y0 - y1)
        c = a + b
        length = math.sqrt(c)

        if length < MAX_LINE_LENGTH_FOR_GAP:
            x0, y0, x1, y1 = scale(x0, x1, y0, y1, SCALING_FACTOR)

        if length > LINE_LENGTH:
            cv2.line(image, (x0, y0), (x1, y1), RED, LINE_THICKNESS, cv2.LINE_AA)

    return image


def detect_horizontal_and_vertical_lines(grayscale_image):
    lsd = cv2.createLineSegmentDetector(0)
    detected_lines = lsd.detect(grayscale_image)

    clone = get_a_white_clone(grayscale_image)

    for detected_line in detected_lines[0]:
        x0 = round(detected_line[0][0])
        y0 = round(detected_line[0][1])
        x1 = round(detected_line[0][2])
        y1 = round(detected_line[0][3])

        # Use slope to determine if line is horizontal or vertical
        # if abs(x1 - x0) < 0.2 or abs((y1 - y0) / (x1 - x0)) < 0.2:
        a = (x0 - x1) * (x0 - x1)
        b = (y0 - y1) * (y0 - y1)
        c = a + b
        length = math.sqrt(c)

        if length < MAX_LINE_LENGTH_FOR_GAP:
            x0, y0, x1, y1 = scale(x0, x1, y0, y1, SCALING_FACTOR)

        if length > LINE_LENGTH:
            cv2.line(clone, (x0, y0), (x1, y1), RED, LINE_THICKNESS, cv2.LINE_AA)

    return clone


def putText(image, text, x, y, _font_size=0.5, _color=(0, 255, 0)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, _font_size, color=_color)
    return image


def get_a_white_clone(image):
    if len(image.shape) == 3:
        return np.ones(image[:, :, 0].shape) * 255  # pure white
    elif len(image.shape) == 2:
        return np.ones(image.shape) * 255  # pure white
    print("Unable to clone image")
    return image


def resize(image, scale):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


def rectangle(image, x, y, w, h, _thickness=2, _color = (0, 255, 0)):
    cv2.rectangle(image, (x, y), (x + w, y + h), thickness=_thickness,
                  color=_color)
