DEBUG = True
RED = [0, 0, 255]
JSON_FILE = "text-recognition-14606bdc50a4.json"
inputs = ["ocr-compatible-fred.png", "delete.jpeg", "delete2.jpeg", "yatharth-0.jpeg"]
INPUT_FILE = "png_image/" + inputs[1]

# Difference in position of two cells for them to be considered in the same row
CELL_POSITION_DIFFERENCE = 10

# TODO: Should probably be wrt cell size
PADDING_ROI_X = 5
PADDING_ROI_Y = 5

MIN_IOU = 0.8

# All lines smaller than this will be scaled by a factor
MAX_LINE_LENGTH_FOR_GAP = 50
SCALING_FACTOR = 2

# -----------PARAMETERS WHICH CAN BE MODIFIED------------
# Threshold for length for a black area to be considered a line (before dilation)
LINE_LENGTH = 3
# For erosion
KERNEL_SIZE = 5
EROSION_ITERATION = 3
# # Minimum area of a cell in table
# CELL_MIN_AREA = 30
APPLY_WEAK_VALIDATION = True
# # Bigger means leniency on cell area ratio
WEAK_VALIDATION_CONSTANT = 6
