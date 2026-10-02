#!/usr/bin/env python
"""
Write a small synthetic DAS archive so the pipeline can be tried without the
(proprietary) field data.

Example
-------
    python scripts/make_synthetic_data.py --out synthetic_archive --minutes 10
    python scripts/run_window.py --data-root synthetic_archive --date 2024/02/02 --hour 0 \
        --first-channel 20 --last-channel 280 --monitor-ch 130 --out results/synthetic --plots
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dastrack.synthetic import make_synthetic_archive


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='synthetic_archive')
    p.add_argument('--start', default='2024-02-02 00:00:00', help='UTC start, "YYYY-MM-DD HH:MM:SS"')
    p.add_argument('--minutes', type=int, default=10)
    p.add_argument('--channels', type=int, default=300)
    p.add_argument('--monitor-ch', type=int, default=150, help='absolute channel all vehicles cross')
    p.add_argument('--vehicles-per-minute', type=float, default=4)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    vehicles = make_synthetic_archive(a.out, start_utc=a.start, minutes=a.minutes,
                                      n_channels=a.channels, monitor_channel=a.monitor_ch,
                                      vehicles_per_minute=a.vehicles_per_minute, seed=a.seed)
    truth = os.path.join(a.out, 'ground_truth.csv')
    with open(truth, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_cross_sec', 'velocity_kmh'])
        for v in vehicles:
            w.writerow([f"{v['t_cross_sec']:.2f}", f"{v['v_mps'] * 3.6:.1f}"])
    print(f"{a.minutes} min, {a.channels} channels, {len(vehicles)} vehicles -> {a.out}  (truth: {truth})")


if __name__ == '__main__':
    main()
