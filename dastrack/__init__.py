"""
dastrack — training-free vehicle tracking on telecom-fiber DAS data by
Kalman-fused F-K slant-stacking.
"""
from .detection import detect_entries
from .fk import fk_candidates, slant_stack
from .io import list_h5_files, load_h5_folder_fast, load_input, load_window
from .kalman import KF
from .params import (BANDPASS, DETECTION, DX, FS, FS_RAW, TRACKER_DEVELOPMENT,
                     TRACKER_PRODUCTION)
from .preprocessing import bandpass_matrix, optimize_preprocessing, preprocess
from .tracker import results_table, run_pipeline, track_vehicle

__version__ = "1.0.0"
