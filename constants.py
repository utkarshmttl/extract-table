DEBUG = True
RED = [0, 0, 255]
JSON_FILE = "text-recognition-14606bdc50a4.json"
INPUT_FILE = "png_image/ocr-compatible-fred.png"

# Difference in position of two cells for them to be considered in the same row
CELL_POSITION_DIFFERENCE = 10

# TODO: Should probably be wrt cell size
PADDING_ROI_X = 5
PADDING_ROI_Y = 5

MIN_IOU = 0.8

# Threshold for length for a black area to be considered a line (before dilation)
LINE_LENGTH = 12
# For erosion
KERNEL_SIZE = 5
# Minimum area of a cell in table
CELL_MIN_AREA = 300
# Bigger means leniency on cell area ratio
WEAK_VALIDATION_CONSTANT = 9
