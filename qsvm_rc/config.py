"""Project-wide constants, paths and section definitions.

Every notebook imports this module, so a value changed here changes it everywhere.
"""

from pathlib import Path

import numpy as np

STUDENT_ID = "25280058"

# Project root is the folder that holds this package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
EXTRA_DATA_ROOT = PROJECT_ROOT / "extra_sections_data"

# One seed constant, used everywhere (NB11 rule)
SEED = 0

# Learning curve protocol from the synopsis
TRAIN_SIZES = [20, 40, 80, 160, 320]
TRAIN_SEEDS = [0, 1, 2, 3, 4]
SAMPLING_STRATEGIES = ["random", "boundary", "mixed"]

# Fixed test set: a random stratified part plus a band of points close to the true boundary
TEST_RANDOM_SIZE = 3000
TEST_BAND_SIZE = 1000
TEST_BAND_HALF_WIDTH = 0.05 # |Pu / (phi Pni) - 1| <= 0.05 defines the boundary band
TRAIN_BAND_HALF_WIDTH = 0.20 # band used by the near-boundary sampling ablation

# Inner cross validation for tuning (same folds for every model)
INNER_FOLDS = 5

# Matched tuning grids: 6 x 7 = 42 settings for both the RBF SVM and the QSVM (widened after the first run hit the grid edges)
SVM_C_GRID = [0.1, 1, 10, 100, 1000, 10000]
RBF_GAMMA_GRID = [0.001, 0.003, 0.01, 0.1, 0.25, 1, 10]
BANDWIDTH_GRID = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1]

# Strength reduction factor used for the labels
PHI = 0.65

# Shot budget for sampled kernels
SHOTS_KERNEL = 2048

# Saved IBM account name from NB02 and the backend named in the plan
IBM_ACCOUNT_NAME = "25280058-key" # 0058-free-ibm
IBM_BACKEND_NAME = "ibm_fez"

# Section definitions. bars_long_face counts the bars along each face parallel to x (corners included),
# bars_short_face counts the bars along each face parallel to y (corners included).
SECTIONS = {
    "3x2": {
        "title": "3 ft x 2 ft (main)",
        "width_x_mm": 914.4,
        "depth_y_mm": 609.6,
        "bars_long_face": 8,
        "bars_short_face": 5,
        "bar_area_mm2": 819,
        "cover_to_centroid_mm": 60,
        "fc_mpa": 30,
        "fy_mpa": 420,
        "dataset_path": PROJECT_ROOT / f"{STUDENT_ID}_AI631_Project_Dataset.csv",
        "extended_path": PROJECT_ROOT / f"{STUDENT_ID}_AI631_Project_Dataset_Extended.csv",
    },
    "3x3": {
        "title": "3 ft x 3 ft (ablation)",
        "width_x_mm": 914.4,
        "depth_y_mm": 914.4,
        "bars_long_face": 9,
        "bars_short_face": 9,
        "bar_area_mm2": 819,
        "cover_to_centroid_mm": 60,
        "fc_mpa": 30,
        "fy_mpa": 420,
        "dataset_path": EXTRA_DATA_ROOT / f"{STUDENT_ID}_AI631_Project_Dataset_3x3.csv",
        "extended_path": EXTRA_DATA_ROOT / f"{STUDENT_ID}_AI631_Project_Dataset_3x3_Extended.csv",
    },
    "5x2_5": {
        "title": "5 ft x 2.5 ft (ablation)",
        "width_x_mm": 1524,
        "depth_y_mm": 762,
        "bars_long_face": 15,
        "bars_short_face": 8,
        "bar_area_mm2": 819,
        "cover_to_centroid_mm": 60,
        "fc_mpa": 30,
        "fy_mpa": 420,
        "dataset_path": EXTRA_DATA_ROOT / f"{STUDENT_ID}_AI631_Project_Dataset_5x2_5.csv",
        "extended_path": EXTRA_DATA_ROOT / f"{STUDENT_ID}_AI631_Project_Dataset_5x2_5_Extended.csv",
    },
}

MAIN_SECTION = "3x2"
FEATURE_COLUMNS = ["pu_kN", "ex_mm", "ey_mm"]


def section_results_dir(section_key, subfolder = None):
    """Folder for one section's results, created on demand."""
    folder = RESULTS_ROOT / section_key
    if subfolder is not None:
        folder = folder / subfolder
    folder.mkdir(parents = True, exist_ok = True)
    return folder


def figures_dir():
    folder = RESULTS_ROOT / "figures"
    folder.mkdir(parents = True, exist_ok = True)
    return folder


def run_seed(base_seed, *offsets):
    """Spaced seeds (SEED + 1000 * a + b ...), so neighbouring runs do not share streams."""
    value = int(base_seed)
    for position, offset in enumerate(offsets):
        value = value + int(offset) * (1000 ** (position + 1))
    return int(np.mod(value, 2 ** 31 - 1))
