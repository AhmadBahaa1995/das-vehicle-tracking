#!/usr/bin/env python
"""
Track every vehicle in one continuous window (e.g. the ten-minute development
window of the manuscript): load -> preprocess -> detect -> F-K + Kalman/RTS.

Example
-------
    python scripts/run_window.py --data-root /path/to/archive \
        --date 2024/02/02 --hour 0 --minutes 0 10 \
        --first-channel <CH0> --last-channel <CH1> --monitor-ch 190 \
        --out results/window --plots

Writes ``vehicles.csv`` (one row per detection; ``valid`` marks the reported
vehicles) and, with ``--plots``, the preprocessing / detection figures and one
diagnostic figure per vehicle.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')

import dastrack as dt
from dastrack import plotting


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', required=True, help='archive root holding YYYY/MM/DD/HH/MM/*.h5')
    p.add_argument('--date', required=True, help='YYYY/MM/DD (UTC folder date)')
    p.add_argument('--hour', type=int, required=True, help='hour folder (UTC)')
    p.add_argument('--minutes', type=int, nargs=2, default=(0, 10), metavar=('START', 'END'),
                   help='minute folders to load, [START, END)  (default 0 10)')
    p.add_argument('--first-channel', type=int, required=True, help='first absolute channel of the slab')
    p.add_argument('--last-channel', type=int, required=True, help='last absolute channel (exclusive)')
    p.add_argument('--monitor-ch', type=int, required=True, help='monitor channel, relative to the slab')
    p.add_argument('--fs', type=float, default=dt.FS, help=f'working sampling rate (default {dt.FS})')
    p.add_argument('--dx', type=float, default=dt.DX, help=f'channel spacing in m (default {dt.DX})')
    p.add_argument('--params', choices=['development', 'production'], default='development',
                   help='tracker parameter set (default: development)')
    p.add_argument('--out', default='results/window', help='output folder')
    p.add_argument('--plots', action='store_true', help='save figures')
    a = p.parse_args()

    folder = os.path.join(a.data_root, a.date, f"{a.hour:02d}")
    raw, fs, t0 = dt.load_window(folder, a.first_channel, a.last_channel,
                                 a.minutes[0], a.minutes[1], target_fs=a.fs)
    if raw is None:
        sys.exit(f"No data loaded from {folder}\n"
                 "To use the sample, download it from Zenodo as described in data/README.md.")
    print(f"Loaded {raw.shape[1]} channels x {raw.shape[0] / fs:.0f} s @ {fs:g} Hz, start {t0}")

    filtered, norm, denoised = dt.preprocess(raw, fs, bandpass=dt.BANDPASS)
    del raw
    os.makedirs(a.out, exist_ok=True)
    plot_dir = os.path.join(a.out, 'figures') if a.plots else None
    if plot_dir:
        plotting.plot_preprocessing(filtered, norm, denoised, fs,
                                    save_path=plotting.out_path(plot_dir, 'preprocessing.png'),
                                    show=False)
    del filtered, norm

    tracker = dt.TRACKER_DEVELOPMENT if a.params == 'development' else dt.TRACKER_PRODUCTION
    _, results, valid = dt.run_pipeline(denoised, fs, a.dx, a.monitor_ch,
                                        tracker_params=tracker, plot_dir=plot_dir,
                                        **dt.DETECTION)

    csv = os.path.join(a.out, 'vehicles.csv')
    dt.results_table(results).to_csv(csv, index=False)
    print(f"\n{len(valid)} valid vehicles -> {csv}")


if __name__ == '__main__':
    main()
