"""
Central configuration: paths, window parameters, label definitions.
Edit this file to adapt the pipeline to different environments or experiments.
"""

import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")      # Directory containing all CSV files
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")        # Directory for saving metrics and plots

# --- Sampling ---
TARGET_HZ = 64                  # Unified resampling frequency (Hz)
WINDOW_STEPS = 128              # Window length in samples at TARGET_HZ (= 2 seconds)
STRIDE_STEPS = 16               # Slide stride (= 1 second, 50% overlap)

# --- Test stride (use a smaller stride on the test set for denser evaluation)
TEST_STRIDE_STEPS = 16         # stride to use when creating test windows (samples)

# --- Training stride for First Fault Label pass
TRAIN_STRIDE_STEPS = 16       # stride for the First Fault Label pass on training files

# --- Label mapping ---
# Merge transition states (15, 16, 17) into label 1 ("recovery transition")
LABEL_REMAP = {15: 1, 16: 1, 17: 1}

# Labels used as classification targets (0-14 after remapping)
VALID_LABELS = list(range(15))

# Human-readable names for each label class
LABEL_NAMES = {
    0:  "Normal",
    1:  "Transition / Not Safe",
    2:  "Area Stop",
    3:  "Light Curtain",
    4:  "Pressure Mat",
    5:  "Door Safety Switch",
    6:  "Emergency Stop",
    7:  "Safety Components Passivated",
    8:  "Position Sensor Broken",
    9:  "RFID Presence Defunct",
    10: "Shuttle Sensor Defunct",
    11: "No Connection ET200S",
    12: "No Connection Control Station",
    13: "No Connection PROFINET",
    14: "No Connection RFID",
}

# --- Columns ---
TIMESTAMP_COLS = ["Year", "Month", "Day", "Hour", "Minute", "Second", "Nanosecond"]
LABEL_COL = "Label"
# Feature columns are resolved dynamically in preprocessing (all columns except timestamps and label)

# --- Train/test split ---
# For each fault class, the file(s) designated as test set (by filename substring match)
# Files not listed here are used for training.
TEST_FILE_MARKERS = {
    0:  ["18-59-13"],               # Normal: last of the 4 unnamed files
    2:  ["2_4_"],
    3:  ["3_4_"],
    4:  ["4_4_"],
    5:  ["5_4_"],
    6:  ["6_4_"],
    7:  ["7_6_"],
    8:  ["8_6_"],
    9:  ["9_4_"],
    10: ["10_4_"],
    11: ["11_4_"],
    12: ["12_4_"],
    13: ["13_4_"],
    14: ["14_4_"],
    1:  [],                         # Label 1 only appears in transition segments; no dedicated test file
}

# --- Class balance ---
MAX_WINDOWS_PER_CLASS = 100  # max windows to keep per class after processing (for data-heavy classes)
# Per-class stride: use smaller stride for minority classes to generate more windows
PER_CLASS_STRIDE = {
    4: 8,   # Pressure Mat
    2: 8,   # Area Stop
    3: 8,   # Light Curtain
    14: 4,  # No Connection RFID (smaller stride to boost recall)
}

# --- Reproducibility ---
RANDOM_SEED = 42
