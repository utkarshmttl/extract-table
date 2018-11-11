import csv

from constants import DEBUG, INPUT_FILE
from image_processing import *
from utils import *

"""
Works only when there is one single table in the PDF
"""

# Minimum area of a cell in table
CELL_MIN_AREA = 220
# Threshold for length for a black area to be considered a line
LINE_LENGTH = 12
# For erosion
KERNEL_SIZE = 5

# Difference in position of two cells for them to be considered in the same row
CELL_POSITION_DIFFERENCE = 10

# TODO: Should probably be wrt cell size
PADDING_ROI_X = 5
PADDING_ROI_Y = 5

MIN_IOU = 0.8


def generate_custom_image(table_list, img_file):
    """
    Generate a custom image to be passed to vision API for OCR
    """
    # Map original image coordinates to generated image coordinates
    mapping = []

    original_image = cv2.imread(img_file)

    # Height and width of custom image. To be initialized later
    height_vision_input = 0
    width_vision_input = -1
    for cell in table_list:
        # Sanity check
        # x = columns, y = rows
        # x y width height
        # Keep increasing height as new cells are added to image
        height_vision_input = height_vision_input + cell[3] + PADDING_ROI_Y * 2
        # Width of new image equals width of largest cell
        if width_vision_input < cell[2]:
            width_vision_input = cell[2]
    new_img = np.ones((height_vision_input + PADDING_ROI_Y, width_vision_input + PADDING_ROI_X * 2, 3)) * 255
    current_y = 0
    w_image = original_image.shape[1]
    h_image = original_image.shape[0]
    for index, cell in enumerate(table_list):
        first = cell[1] if cell[1] - PADDING_ROI_Y < 0 else cell[1] - PADDING_ROI_Y
        second = h_image - 1 if cell[1] + cell[3] + PADDING_ROI_Y >= h_image else cell[1] + cell[3] + PADDING_ROI_Y
        third = cell[0] if cell[0] - PADDING_ROI_X < 0 else cell[0] - PADDING_ROI_X
        fourth = w_image - 1 if cell[0] + cell[2] + PADDING_ROI_X >= w_image else cell[0] + cell[2] + PADDING_ROI_X

        new_img[current_y: current_y + second - first, 0: fourth - third, :] = original_image[first:second,
                                                                               third:fourth, :]
        mapping.append({})
        mapping[index]["original_contours"] = [first, second, third, fourth]
        mapping[index]["new_contours"] = [current_y, current_y + second - first, 0, fourth - third]
        current_y += second - first
    cv2.imwrite("4_new_image.png", new_img)

    text_results = detect_text("4_new_image.png")
    # print(text_results)

    # LIST CONTAINING [TEXT, LEFT, TOP, RIGHT, BOTTOM] FOR EACH BOUNDING BOX
    extracted_info = get_left_top_right_bottom(text_results)

    processed = cv2.imread("4_new_image.png")
    for info in extracted_info:
        cv2.rectangle(processed, (info[1], info[2]), (info[3], info[4]), thickness=2, color=(0, 255, 0))
        cv2.putText(processed, info[0], (info[1], info[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
    cv2.imwrite("5_image_with_text.png", processed)

    # for info in extracted_info:
    #     print(info)


def get_table_cells(image):
    # List which will save each cell's coordinates
    ret_list = []
    # Read image using opencv
    img = cv2.imread(image)
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect horizontal and vertical line in image
    lined_image = detect_horizontal_and_vertical_lines(gray, img)

    if DEBUG:
        cv2.imwrite("1_lsd_result_" + str(LINE_LENGTH) + ".png", lined_image)

    # Create a white image of same dimensions as input
    gray_version = get_a_white_clone(img)

    # Extract red pixels from image (Red represents detected lines drawn on the image
    raw_lines = extract_red_from_image(gray_version, lined_image)

    # kernel is used for erode operation
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    # this is actually dilation (and not erosion) because b/w are reversed in our case
    eroded_image = cv2.erode(raw_lines, kernel, iterations=1)

    if DEBUG:
        cv2.imwrite("2_erode_result.png", eroded_image)

    # Find contours in eroded image with detected lines
    _, contours, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area
    sorted_contours = sorted(contours, key=lambda _x: cv2.contourArea(_x))

    # TODO: Only works for single table per image and assumes that table is second largest contour
    table_contour = sorted_contours[-2]
    # Get dimensions of rectangle around contour
    x_t, y_t, w_t, h_t = cv2.boundingRect(table_contour)

    white_clone = get_a_white_clone(img)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        curr_area = cv2.contourArea(cnt)
        # Check if current contour is within table
        if x_t < x < x_t + w_t and y_t < y < y_t + h_t and curr_area > CELL_MIN_AREA:
            inner_list = [x, y, w, h]
            cv2.rectangle(white_clone, (x, y), (x + w, y + h), thickness=2, color=(0, 0, 0))
            ret_list.append(inner_list)

    cv2.imwrite("3_table_contours.png", white_clone)
    print(white_clone.shape, img.shape)

    return ret_list, img


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

    area_list = np.ones(len(cells))
    k = 0
    for cell in cells:
        area_list[k] = cell[2] * cell[3]
        k = k + 1

    cells = np.array(cells)
    idxs = np.argsort(area_list)
    cells = cells[idxs]
    area_list = np.sort(area_list)
    i = 0
    prev_ratio = -1
    index = 0
    while i + 1 < len(area_list):
        ratio = area_list[i + 1] / area_list[i]
        if ratio > prev_ratio:
            prev_ratio = ratio
            index = i
        i = i + 1

    if prev_ratio < 6:
        return cells

    cells = cells[index + 1:length, :].tolist()

    image = get_a_white_clone(image)
    for cell in cells:
        cv2.rectangle(image, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), thickness=2, color=(0, 255, 0))
    cv2.imwrite("4_cleaned_table.png", image)

    return cells


def compare_ocr_with_cell_bboxes(cells, extracted_info):
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


def main(file_to_read):
    # pass current image to Google OCR
    text_results = detect_text(file_to_read)

    extracted_info = get_left_top_right_bottom(text_results)

    # Locate table in image and extract individual cells
    cells, clone = get_table_cells(file_to_read)  # [[x, y, w, h], [x, y, w, h], ...]
    # pre-processing on table cells
    cells = weak_validation(cells, clone)

    cells_with_text = compare_ocr_with_cell_bboxes(cells, extracted_info)

    # sort cells by their top-y coordinate
    cells_with_text.sort(key=lambda x: x[1])

    # LIST WILL CONTAIN LISTS OF TEXTS WHICH ARE ON SAME LINES
    table = sort_list(cells_with_text)

    write_table_to_file("output.csv", table)


main(INPUT_FILE)
