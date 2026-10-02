#!/usr/bin/env python
"""
Per-vehicle diagnostics drawn at random from across the production period,
one per week (manuscript Figure 9).

For each week: pick a random (day, hour), load a short window, track every
detection with the production parameters, choose one valid vehicle at random
and draw its six-panel diagnostic with the other tracked vehicles faded behind.

Example
-------
    python scripts/plot_weekly_diagnostics.py --data-root /path/to/archive \
        --start 2024-02-01 --end 2024-02-28 \
        --first-channel <CH0> --last-channel <CH1> --monitor-ch 190 \
        --seed 0 --out results/month/weekly_diagnostics
"""
import argparse
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')

import dastrack as dt
from dastrack import plotting


def weeks_between(start, end):
    d0 = datetime.strptime(start, '%Y-%m-%d')
    d1 = datetime.strptime(end, '%Y-%m-%d')
    weeks, cur = [], d0
    while cur <= d1:
        wk_end = min(cur + timedelta(days=6), d1)
        weeks.append((cur, wk_end))
        cur = wk_end + timedelta(days=1)
    return weeks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', required=True)
    p.add_argument('--start', required=True, help='YYYY-MM-DD')
    p.add_argument('--end', required=True, help='YYYY-MM-DD (inclusive)')
    p.add_argument('--first-channel', type=int, required=True)
    p.add_argument('--last-channel', type=int, required=True)
    p.add_argument('--monitor-ch', type=int, required=True)
    p.add_argument('--fs', type=float, default=dt.FS)
    p.add_argument('--dx', type=float, default=dt.DX)
    p.add_argument('--minutes', type=int, default=10, help='window length per draw (min)')
    p.add_argument('--tries', type=int, default=6, help='random draws per week before giving up')
    p.add_argument('--per-week', type=int, default=1, help='vehicles to draw per week')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='weekly_diagnostics')
    a = p.parse_args()

    rng = random.Random(a.seed)
    det = dt.DETECTION
    for wk_i, (w0, w1) in enumerate(weeks_between(a.start, a.end), start=1):
        for draw in range(a.per_week):
            for attempt in range(a.tries):
                day = w0 + timedelta(days=rng.randint(0, (w1 - w0).days))
                hour = rng.randint(0, 23)
                date_str = day.strftime('%Y/%m/%d')
                print(f"[week {wk_i}] {date_str} {hour:02d}:00 (attempt {attempt + 1}/{a.tries})")
                raw, fs, _ = dt.load_window(os.path.join(a.data_root, date_str, f"{hour:02d}"),
                                            a.first_channel, a.last_channel, 0, a.minutes,
                                            target_fs=a.fs, verbose=False)
                if raw is None:
                    print("   no data, retrying")
                    continue
                _, _, denoised = dt.preprocess(raw, fs, bandpass=dt.BANDPASS, verbose=False)
                del raw
                starts, _ = dt.detect_entries(denoised, fs, monitor_channel=a.monitor_ch,
                                              min_height=det['min_height'],
                                              min_distance_sec=det['min_dist_sec'], verbose=False)
                valid = []
                for t0 in starts:
                    r = dt.track_vehicle(t0, denoised, fs, a.dx, a.monitor_ch,
                                         v_min_kmh=det['v_min_kmh'], v_max_kmh=det['v_max_kmh'],
                                         **dt.TRACKER_PRODUCTION)
                    if r and r['valid']:
                        valid.append(r)
                if not valid:
                    print(f"   {len(starts)} detection(s), none valid, retrying")
                    continue
                chosen = rng.choice(valid)
                label = f"wk{wk_i}_{date_str.replace('/', '-')}_{hour:02d}h_t{chosen['t_sec']:.0f}s"
                chosen['car_id'] = label
                background = [plotting.track_from_result(r) for r in valid if r is not chosen]
                plotting.plot_vehicle_diagnostic(
                    chosen, denoised, fs, a.dx, background_tracks=[b for b in background if b],
                    save_path=plotting.out_path(a.out, f"diagnostic_{label}.png"), show=False)
                print(f"   {len(valid)}/{len(starts)} valid; plotted {label}")
                break
            else:
                print(f"[week {wk_i}] no valid vehicle found in {a.tries} tries")


if __name__ == '__main__':
    main()
