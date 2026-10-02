#!/usr/bin/env python
"""
Month-scale production run: every hour in a date range, in parallel, each hour
split into continuous 10-minute chunks.  Writes one catalog row per vehicle.

Example (the February 2024 run of the manuscript)
-------------------------------------------------
    python scripts/run_month.py --data-root /path/to/archive \
        --start 2024-02-01 --end 2024-02-28 \
        --first-channel <CH0> --last-channel <CH1> --monitor-ch 190 \
        --workers 8 --out results/month

Outputs in ``--out``:
    catalog_<start>_to_<end>.csv   one row per vehicle (time in local time, JST by default)
    qc/<YYYY-MM-DD>/Chunk_HH-MM.png  per-chunk waterfall with all tracks (unless --no-qc)
    run.log                        per-worker log
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')

import dastrack as dt
from dastrack import production


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', required=True, help='archive root holding YYYY/MM/DD/HH/MM/*.h5')
    p.add_argument('--start', required=True, help='first day, YYYY-MM-DD (UTC folder dates)')
    p.add_argument('--end', required=True, help='last day (inclusive), YYYY-MM-DD')
    p.add_argument('--first-channel', type=int, required=True, help='first absolute channel of the slab')
    p.add_argument('--last-channel', type=int, required=True, help='last absolute channel (exclusive)')
    p.add_argument('--monitor-ch', type=int, required=True, help='monitor channel, relative to the slab')
    p.add_argument('--fs-raw', type=float, default=dt.FS_RAW, help=f'native sampling rate (default {dt.FS_RAW:g})')
    p.add_argument('--fs-target', type=float, default=dt.FS, help=f'working sampling rate (default {dt.FS})')
    p.add_argument('--dx', type=float, default=dt.DX, help=f'channel spacing in m (default {dt.DX})')
    p.add_argument('--files-per-chunk', type=int, default=20,
                   help='HDF5 segments per chunk (default 20 = 10 min of 30-s files)')
    p.add_argument('--utc-offset', type=float, default=9, help='hours added to UTC in the catalog (default 9, JST)')
    p.add_argument('--workers', type=int, default=8,
                   help='worker processes (default 8; bounded in practice by storage I/O)')
    p.add_argument('--no-qc', action='store_true', help='do not write per-chunk QC images')
    p.add_argument('--out', default='results/month', help='output folder')
    a = p.parse_args()

    if not os.path.isdir(a.data_root):
        sys.exit(f"Data root not found: {a.data_root}")
    if not 0 <= a.monitor_ch < a.last_channel - a.first_channel:
        sys.exit("Monitor channel lies outside the loaded slab.")
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(a.out, 'run.log'))])

    n_hours = len(production.build_tasks(a.start, a.end))
    print(f"{a.start} -> {a.end}: {n_hours} hours | channels {a.first_channel}-{a.last_channel} "
          f"| monitor {a.monitor_ch} (relative) | {a.workers} workers")
    t_wall = time.time()
    records = production.run_production(
        a.start, a.end, max_workers=a.workers,
        data_root=a.data_root, dx=a.dx, fs_raw=a.fs_raw, fs_target=a.fs_target,
        monitor_ch=a.monitor_ch, ch_start=a.first_channel, ch_end=a.last_channel,
        detection=dt.DETECTION, tracker_params=dt.TRACKER_PRODUCTION,
        files_per_chunk=a.files_per_chunk,
        plot_dir=None if a.no_qc else os.path.join(a.out, 'qc'))
    elapsed = time.time() - t_wall

    csv = os.path.join(a.out, f"catalog_{a.start}_to_{a.end}.csv")
    df = production.write_catalog(records, csv, utc_offset_hours=a.utc_offset)
    print(f"\n{len(df)} vehicles -> {csv}")
    if len(df):
        n_pos = int((df['speed'] > 0).sum())
        print(f"  direction + / - : {n_pos} / {len(df) - n_pos}"
              f" | mean speed {df['speed'].abs().mean():.1f} km/h"
              f" | median {df['speed'].abs().median():.1f} km/h")
    print(f"  wall time {elapsed / 60:.1f} min ({elapsed / max(n_hours, 1):.0f} s per hour of data)")


if __name__ == '__main__':
    main()
