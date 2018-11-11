import io
import os
import time

from google.cloud import vision

from constants import JSON_FILE

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = JSON_FILE

# How many non-ascii characters have been detected in a cell
SPECIAL_SYMBOL_THRESHOLD = 2

THRESHOLD_DIFF_FOR_SAME_LINE = 15


def get_left_top_right_bottom(texts):
    extracted_info = []
    for text in texts:
        bound = text.bounding_poly
        # LEFTMOST POSITION OF BOUNDING BOX
        min_x = min([vertex.x for vertex in bound.vertices])
        # TOPMOST POSITION OF BOUNDING BOX
        min_y = min([vertex.y for vertex in bound.vertices])
        # RIGHTMOST POSITION OF BOUNDING BOX
        max_x = max([vertex.x for vertex in bound.vertices])
        # BOTTOMMOST POSITION OF BOUNDING BOX
        max_y = max([vertex.y for vertex in bound.vertices])
        extracted_info.append([text.description, min_x, min_y, max_x, max_y])
    return extracted_info


def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    try:
        response = client.text_detection(image=image)
    except:
        print("Google Error Occured")
        time.sleep(10)
        detect_text(path)
    return response.text_annotations[1:]


def preprocess(cell):
    """
    Preprocess cell for numeric as well as non-ascii content
    :param cell:
    :return:
    """
    alphanumeric_comparewith = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789., ()"
    comparewith = "0123456789"
    if len(cell) == 0:
        return cell
    cell = cell.replace("\n", " ")

    if cell[-2:] == "00":
        for index, letter in enumerate(cell):
            if letter not in comparewith:
                cell = cell.replace(letter, '')
        cell = cell[:-2] + "." + cell[-2:]
        return cell
    count = 0
    for character in cell:
        if character not in alphanumeric_comparewith:
            count += 1
    if count > SPECIAL_SYMBOL_THRESHOLD:
        return ""
    return cell


def sort_list(extracted_info):
    final_extracted_info = []
    # TEMP VARIABLES
    prev_y = -1
    new_extracted_info = []
    # APPENDS ALL TEXTS ON SAME LINE INTO ONE LIST
    for info in extracted_info:
        if info[1] - prev_y > THRESHOLD_DIFF_FOR_SAME_LINE and prev_y != -1:
            final_extracted_info.append(new_extracted_info)
            new_extracted_info = [info]
            prev_y = info[1]
        else:
            new_extracted_info.append(info)
            prev_y = info[1]
    return final_extracted_info


def get_iou(bbox1, bbox2):
    """
    Computes Intersection-over-Union between two given bounding boxes

    :param bbox1: [L, T, R, B]
    :param bbox2: [L, T, R, B]
    :return: (Float) IOU between the two bounding boxes
    """
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    boxBArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_intersection_ratio(bbox1, bbox2):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    boxBArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    if boxAArea > boxBArea:
        iou = interArea / float(boxBArea)
    else:
        iou = interArea / float(boxAArea)
    return iou
