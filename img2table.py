import csv

import pytesseract
from PIL import Image
from scipy.stats import mode

from constants import INPUT_FILE, PADDING_ROI_Y, PADDING_ROI_X, EROSION_ITERATION, WEAK_VALIDATION_CONSTANT, \
    APPLY_WEAK_VALIDATION, KERNEL_SIZE, MIN_IOU
from image_processing import *
from utils import *

"""
Works only when there is one single table in the PDF
"""


def get_table_cells(gray):
    """
    Given an image, look for the largest table and extract its cells
    :param gray: Grayscale image
    :return: List of lists containing table cell BBoxes
    """

    # List which will save each cell's coordinates
    ret_list = []
    # Temporary list required to process contours
    hierarchy_list = []

    # Detect horizontal and vertical line in image
    lined_image = detect_horizontal_and_vertical_lines(gray)

    cv2.imwrite("1_lsd_result_" + str(LINE_LENGTH) + ".png", lined_image)

    # Extract detected lines from image
    raw_lines = extract_lines_from_image(lined_image)

    # kernel is used for erode operation
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    # this is actually dilation (and not erosion) because b/w are reversed in our case
    eroded_image = cv2.erode(raw_lines, kernel, iterations=EROSION_ITERATION)

    cv2.imwrite("2_erode_result.png", eroded_image)

    # hierarchy = [Next, Previous, First_Child, Parent]
    # Find contours in eroded image with detected lines
    _, contours, hierarchy = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area
    sorted_contours = sorted(contours, key=lambda _x: cv2.contourArea(_x))

    # TODO: Only works for single table per image and assumes that table is second largest contour
    table_contour = sorted_contours[-2]
    # Get dimensions of rectangle around contour
    x_t, y_t, w_t, h_t = cv2.boundingRect(table_contour)

    white_clone = get_a_white_clone(gray)
    for index, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Check if current contour is within table
        if x_t < x < x_t + w_t and y_t < y < y_t + h_t:
            inner_list = [x, y, w, h]
            cv2.rectangle(white_clone, (x, y), (x + w, y + h), thickness=2, color=(0, 0, 0))
            ret_list.append(inner_list)
            hierarchy_list.append(hierarchy[0][index])

    cv2.imwrite("3_table_contours.png", white_clone)

    new_ret_list = []

    # To remove small BBoxes, we check who is the parent of current contour.
    # Most contours will have a common parent
    all_parent = mode(hierarchy_list, axis=0).mode[0][3]

    for index, h in enumerate(hierarchy_list):
        if h[3] == all_parent:
            new_ret_list.append(ret_list[index])

    return new_ret_list


def weak_validation(cells, image):
    """
    Expects cells list and return cells list,
    removes very small contours, based on ratio of area. Where ratio
    between area of contour is maximum, that is where we can think
    noise contours start
    """
    length = len(cells)

    if length < 1:
        return

    # All cells' area will be stored here
    area_list = np.ones(len(cells))
    k = 0
    # Store cell area
    for cell in cells:
        area_list[k] = cell[2] * cell[3]
        k = k + 1

    cells = np.array(cells)
    idxs = np.argsort(area_list)

    cells = cells[idxs]
    # Sorted by area
    area_list = np.sort(area_list)
    i = 0
    prev_ratio = -1
    index = 0
    # Store index where nearby elements have largest area difference ratio
    while i + 1 < len(area_list):
        ratio = area_list[i + 1] / area_list[i]
        if ratio > prev_ratio:
            prev_ratio = ratio
            index = i
        i = i + 1

    if prev_ratio < WEAK_VALIDATION_CONSTANT:
        print("ALL ARE GOOD")

        for cell in cells:
            rectangle(image, cell[0], cell[1], cell[2], cell[3])
        cv2.imwrite("4_cleaned_table.png", image)
        return cells.tolist()

    print(index + 1, " BOXES REMOVED")
    cells = cells[index + 1:length, :].tolist()

    for cell in cells:
        rectangle(image, cell[0], cell[1], cell[2], cell[3])
    cv2.imwrite("4_cleaned_table.png", image)

    return cells


def compare_ocr_with_cell_bboxes(image, cells, extracted_info):
    """
    Compare IoU of Vision BBoxes with extracted cells BBoxes
    :return cells with all the text inside its perimeter [x, y, w, h, text]
    """
    for index, cell in enumerate(cells):
        for info in extracted_info:
            iou = get_intersection_ratio(info[1:], [cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[3]])

            if iou > MIN_IOU:
                if len(cell) == 4:
                    cells[index].append([info[0]])
                else:
                    cells[index][4].append(info[0])
        if len(cells[index]) == 4:
            cells[index].append("")
            part_of_image = image.crop((cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[3]))
            data = pytesseract.image_to_string(part_of_image, config="-psm 6")
            cells[index][4] = data
        else:
            cells[index][4] = " ".join(cells[index][4])

    return cells


def write_table_to_file(name_of_file, table):
    with open(name_of_file, 'w+', encoding='utf8') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        rows_to_write = []

        for row in table:
            row.sort(key=lambda x: x[0])
            print(row)

            new_row = []
            for cell in row:
                new_row.append(cell[4])
            rows_to_write.append(new_row)
        writer.writerows(rows_to_write)


def get_coords_with_padding(image, cell):
    h_image = image.shape[0]
    w_image = image.shape[1]
    first = cell[1] if cell[1] - PADDING_ROI_Y < 0 else cell[1] - PADDING_ROI_Y
    first_opp = cell[1] + PADDING_ROI_Y / 2
    second = h_image - 1 if cell[3] + PADDING_ROI_Y >= h_image else cell[3] + PADDING_ROI_Y
    second_opp = cell[3] - PADDING_ROI_Y / 2
    third = cell[0] if cell[0] - PADDING_ROI_X < 0 else cell[0] - PADDING_ROI_X
    fourth = w_image - 1 if cell[2] + PADDING_ROI_X >= w_image else cell[2] + PADDING_ROI_X
    # return third, first, fourth, second
    return third, int(first_opp), fourth, int(second_opp)


def apply_white_patch_over_text(extracted_info, clone):
    for info in extracted_info:
        x1, y1, x2, y2 = get_coords_with_padding(clone, info[1:])
        cv2.rectangle(clone, (x1, y1), (x2, y2), thickness=cv2.FILLED,
                      color=(255, 255, 255))
    return clone


def main(file_to_read):
    gray = convert_to_grayscale(file_to_read)
    cv2.imwrite("0_grayscale.png", gray)

    # pass current image to Google OCR
    text_results = detect_text_in_document(file_to_read)

    # Applying white patch over text helps clean table cells better
    clone = apply_white_patch_over_text(text_results, gray)
    # Locate table in image and extract individual cells
    cells = get_table_cells(clone)  # [[x, y, w, h], [x, y, w, h], ...]

    #
    white_clone = get_a_white_clone(clone)
    if APPLY_WEAK_VALIDATION:
        # pre-processing on table cells to remove small contours
        cells = weak_validation(cells, white_clone.copy())

    for cell in cells:
        rectangle(white_clone, cell[0], cell[1], cell[2], cell[3])
    cv2.imwrite("4_cleaned_table.png", white_clone)

    cells_with_text = compare_ocr_with_cell_bboxes(Image.open(file_to_read), cells, text_results)
    # sort cells by their top-y coordinate
    cells_with_text.sort(key=lambda x: x[1])
    #
    # LIST WILL CONTAIN LISTS OF TEXTS WHICH ARE ON SAME LINES
    table = sort_list(cells_with_text)
    #
    write_table_to_file("output.csv", table)


main(INPUT_FILE)
