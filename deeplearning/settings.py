from pathlib import Path

# ============================================================
# Project paths
# ============================================================

BASE_PATH = Path(r"D:\Projects\welding-droplet-vision")
DATA_PATH = Path(r"D:\Projects\welding-droplet-vision-data")

VIDEO_PATH = DATA_PATH / "videos"
FRAME_PATH = DATA_PATH / "frames"
SAMPLE_PATH = DATA_PATH / "sampled_frames"
DATASET_PATH = DATA_PATH / "dataset"
OUTPUTS_PATH = DATA_PATH / "outputs"

SAMPLE_RECORDS_PATH = SAMPLE_PATH / "sampling_records.csv"
C_LABEL_PATH = DATASET_PATH / "classification_labels.csv"
R_LABEL_PATH = DATASET_PATH / "regression_labels.csv"


# ============================================================
# Video dataset
# ============================================================
# filename   - Video filename
# name       - Video identifier
# frame_num  - Total number of frames
# period_num - Number of droplet transfer cycles (determined by manual video inspection)
VIDEOS_INFO = [
    {"filename": "20_.avi", "name": "20", "frame_num": 3000, "period_num": 20},
    {"filename": "21_.avi", "name": "21", "frame_num": 3000, "period_num": 20},
    {"filename": "22_.avi", "name": "22", "frame_num": 3001, "period_num": 1},
    {"filename": "24__clip.wmv", "name": "24c", "frame_num": 1036, "period_num": 14},
    {"filename": "26__clip.wmv", "name": "26c", "frame_num": 737, "period_num": 3},
    {"filename": "29_.avi", "name": "29", "frame_num": 3000, "period_num": 60},
    {"filename": "Y4 Camera33_.avi", "name": "33", "frame_num": 300, "period_num": 9},
]


# ============================================================
# Annotation and detection
# ============================================================
WINDOW_SIZE = 32
ZOOM_FACTOR = 5
DISPLAY_SIZE = WINDOW_SIZE * ZOOM_FACTOR

C_LABELS = {"full": 1, "upper": 2, "lower": 3, "off": 0}
BOUNDARY_POINTS = ["upper_left", "upper_right", "bottom", "left", "right"]
R_WINDOWS = ["full", "upper", "lower"]


# ============================================================
# Dataset construction
# ============================================================

DEFAULT_FRAME_TEST_RATIO = 1 / 6
RANDOM_SEED = 42

# Training hyperparameters recorded in the thesis.
C_BATCH_SIZE = 100
R_BATCH_SIZE = 80

# Dataset split sizes recorded in thesis Table 3-3.
# Classification: each class has 4,000 crops (3,000 train + 1,000 test).
C_TRAIN_FRAMES = 3000
C_TEST_FRAMES = 1000

# Regression: target_id -> (train sample count, test sample count).
# These are per-target sample counts, not source-frame counts.
R_SPLIT_COUNTS = {
    0: (9200, 400),   # upper_left: total 9,600
    1: (9000, 100),   # upper_right: total 9,100
    2: (7680, 240),   # bottom: total 7,920
    3: (10560, 400),  # left: total 10,960
    4: (10880, 80),   # right: total 10,960
}

# "sample" reproduces the historical sequential split (first N samples train, following M samples test).
# "frame" prevents samples from the same source frame appearing in both splits,
# but cannot generally reproduce exact table counts.
R_SPLIT_MODE = "sample"


# ============================================================
# Display
# ============================================================

# Colors are in BGR format.
TARGET_COLORS = {
    0: (0, 0, 255),    # upper_left: red
    1: (0, 165, 255),  # upper_right: orange
    2: (0, 255, 255),  # bottom: yellow
    3: (255, 0, 0),    # left: blue
    4: (255, 0, 255),  # right: magenta
}
