"""
Parameter sets.

``DETECTION`` holds the detection / validity settings shared by every run.

``TRACKER_DEVELOPMENT`` is the tracker configuration used on the ten-minute
development window (manuscript Method section, Figures 3-5 and 7).

``TRACKER_PRODUCTION`` is the configuration used for the month-scale
February 2024 production run (manuscript "Month-Scale Production Deployment",
Figures 8-9, Table 4).  It differs from the development set only in the
velocity-gate limits and the Kalman noise terms (smoother tracks).
"""

# Acquisition
DX = 2.0419             # channel spacing (m); every reported speed scales linearly with it
FS_RAW = 1000.0         # interrogator sampling rate (Hz)
FS = 250                # working sampling rate after decimation (Hz)
BANDPASS = (70, 100)    # vehicle tyre/engine band (Hz)

# Detection at the monitor channel + validity bounds (shared)
DETECTION = dict(
    min_height=0.30,        # threshold on the smoothed monitor envelope (0-1)
    min_dist_sec=4.0,       # refractory interval between two detections (s)
    v_min_kmh=10,           # slower than this -> rejected
    v_max_kmh=80,           # faster than this -> rejected
)

_TRACKER_COMMON = dict(
    # bootstrap F-K patch (proposes the seed velocities)
    kf_bootstrap_half_time=5.0,     # +/- s of the initial wide patch
    kf_bootstrap_half_ch=90,        # +/- channels of that patch

    # sliding tracker window (one F-K measurement per window)
    kf_window_sec=2.0,              # +/- s per window
    kf_half_space_ch=20,            # +/- channels per window
    kf_slide_overlap=0.5,           # window overlap
    env_rms_threshold=0.03,         # skip windows fainter than this RMS

    # measurement gating
    gate_sigma=4.0,                 # x (KF velocity uncertainty)

    # alive / exit tests
    min_contrast=1.5,               # window alive only if F-K peak/median >= this
    max_consecutive_misses=2,       # end track after this many dead windows
    max_ch_jump=13.0,               # de-slant centroid trust radius (channels)
    min_local_energy=0.004,         # pixel-energy floor at the predicted channel

    # detection anchor (keeps two close vehicles from sharing an anchor)
    anchor_half_ch=1,
    anchor_half_t_sec=0.5,
    anchor_max_shift=1,

    # measurement-noise scaling with F-K contrast; position skepticism
    contrast_r_lo=0.3,
    contrast_r_hi=8.0,
    pos_skeptic=3.0,

    # self-rejection of degenerate tracks
    min_track_updates=3,
    min_track_dist_m=10.0,

    # track extent + multi-seed bootstrap
    track_back_max_sec=20.0,
    track_forward_max_sec=20.0,
    n_seeds=3,
)

TRACKER_DEVELOPMENT = dict(
    _TRACKER_COMMON,
    gate_lo_kmh=25.0,               # gate never tighter than this
    gate_hi_kmh=35.0,               # gate never wider than this
    q_vel=0.8,                      # KF process noise
    r_vel=6.25,                     # KF velocity measurement noise
)

TRACKER_PRODUCTION = dict(
    _TRACKER_COMMON,
    gate_lo_kmh=20.0,
    gate_hi_kmh=45.0,
    q_vel=0.4,
    r_vel=9.0,
)
