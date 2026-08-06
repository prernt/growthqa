from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_META_CSV = ROOT / "data" / "train_data" / "training_meta.csv"
MODEL_DIR = ROOT / "classifier_output" / "saved_models_selected"
LOCKFILE_OUT = ROOT / "classifier_output" / "requirements_lock.txt"
RANDOM_STATE = 42
STEP_HOURS = 0.5
TMAX_HOURS = 16.0
MIN_POINTS = 4
SMOOTH_METHOD = "SGF"
SMOOTH_WINDOW = 5
NORMALIZE = "MINMAX"
TRUNC_HORIZONS = [8.0, 10.0, 12.0, 14.75, 16.0]
TRUNC_PER_CURVE = 3
TRUNC_SEED = 123
GAP_AUG_FRACTION = 0.30       
GAP_AUG_PER_CURVE = 2         
GAP_AUG_SEED = 456
GAP_MIN_HOURS = 2.0           
GAP_MAX_HOURS = 6.0
GAP_MIN_MISSING_FRAC = 0.40   
GAP_MAX_MISSING_FRAC = 0.80
MAX_GAP_HOURS_OVERRIDE = 10.0
MISSING_FRAC_OVERRIDE = 0.85
MISSING_FEATURE_FRAC_OVERRIDE = 0.25
STAGE1_FEATURE_GROUPS = {
    "observation_quality": [
        "observed_tmax",            # how much of the curve was actually observed
        "n_points_observed",        # raw point count: data density (raw-data-based, not grid-based)
        "max_gap_hours",            # largest real gap between measurements (raw-data-based)
        "missing_frac_on_grid",     # measurement density relative to the canonical grid (raw-data-based)
    ],
    "level": [
        "initial_OD",               # starting level
        "final_OD",                 # ending level
    ],
    "growth_dynamics": [
        "net_change_per_hour",      # average rate over the whole window
        "max_slope",                # peak instantaneous rate
        "auc_per_hour",             # average level over time (distinct from rate)
        "lag_time_est",             # onset of active growth
        "growth_phase_duration",    # duration of the active growth phase
    ],
    "shape_integrity": [
        "monotonicity_fraction",    # overall directional consistency
        "largest_drop_frac",        # worst single decline
        "multi_phase_flag",         # diauxic / double-peak detector
        "roughness",                # raw jaggedness (includes trend)
        "noise_residual_std",       # noise after removing trend (isolates noise alone)
    ],
}
STAGE1_CANDIDATE_POOL = [f for group in STAGE1_FEATURE_GROUPS.values() for f in group]
STAGE1_TOP_10_FEATURES = [
    "largest_drop_frac",       # shape_integrity   (rank 1,  importance 0.1071)
    "auc_per_hour",            # growth_dynamics   (rank 2,  importance 0.0185)
    "growth_phase_duration",   # growth_dynamics   (rank 3,  importance 0.0177)
    "multi_phase_flag",        # shape_integrity   (rank 4,  importance 0.0168)
    "max_slope",               # growth_dynamics   (rank 5,  importance 0.0133)
    "final_OD",                # level             (rank 6,  importance 0.0053)
    "net_change_per_hour",     # growth_dynamics   (rank 7,  importance 0.0041)
    "lag_time_est",            # growth_dynamics   (rank 8,  importance 0.0039)
    "roughness",               # shape_integrity   (rank 9,  importance 0.0038)
    "max_gap_hours",           # observation_quality (rank 10, importance 0.0034)
]
STAGE1_TOP_8_FEATURES = [
    "largest_drop_frac",       # shape_integrity
    "auc_per_hour",            # growth_dynamics
    "growth_phase_duration",   # growth_dynamics
    "multi_phase_flag",        # shape_integrity
    "max_slope",               # growth_dynamics
    "final_OD",                # level
    "net_change_per_hour",     # growth_dynamics
    "max_gap_hours",           # observation_quality (swapped in for lag_time_est)
]
STAGE1_SELECTED_FEATURES = STAGE1_TOP_10_FEATURES
IDENTIFIER_COLS = {
    "FileName",
    "Test Id",
    "Model Name",
    "Concentration",
    "base_curve_id",
    "aug_id",
    "tmax_original",
    "train_horizon",
    "is_synthetic",
    "is_censored",
    "gap_augmented",
    "gap_pattern",
}
LEAKAGE_COLS = {"best_model_name"}
LATE_WINDOW_REFERENCE_STEP_HOURS = 1.0
LATE_WINDOW_MAX_MISSING_FRAC = 0.85
MIN_LATE_POINTS_FLOOR = 3
MIN_LATE_POINTS_CEILING = 10
MIN_LATE_HOURS_ANCHOR = 2.5
MIN_LATE_POINTS_FALLBACK_RATE_PER_HOUR = 2.0
