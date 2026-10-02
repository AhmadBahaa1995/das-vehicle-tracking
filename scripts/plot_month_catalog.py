#!/usr/bin/env python
"""
Summary statistics and the five-panel catalog figure (manuscript Figure 8)
from the CSV written by ``run_month.py``.

Example
-------
    python scripts/plot_month_catalog.py \
        --csv results/month/catalog_2024-02-01_to_2024-02-28.csv \
        --start 2024-02-01 --end 2024-02-28 --out results/month/figure8_catalog.png

``--start/--end`` (local dates, inclusive) drop the partial boundary day
created by the UTC -> JST shift.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')

from dastrack import catalog


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True)
    p.add_argument('--start', help='first local date to keep, YYYY-MM-DD')
    p.add_argument('--end', help='last local date to keep (inclusive), YYYY-MM-DD')
    p.add_argument('--out', default='figure8_catalog.png')
    a = p.parse_args()

    df = catalog.load_catalog(a.csv, a.start, a.end)
    if df.empty:
        sys.exit("No vehicles in the selected range.")
    s = catalog.summarize_catalog(df)
    print(f"Vehicles                 : {s['vehicles']:,}  (~{s['mean_rate_per_hour']:.0f} per hour)")
    print(f"Mean speed  + / -        : {s['mean_speed_right_kmh']:.1f} / {s['mean_speed_left_kmh']:.1f} km/h")
    print(f"Directional split  + / - : {s['n_right']:,} / {s['n_left']:,}"
          f"  ({100 * s['n_right'] / s['vehicles']:.0f} / {100 * s['n_left'] / s['vehicles']:.0f} %)")
    print(f"Weekday vehicles/day     : {s['weekday_daily_min']:,.0f}-{s['weekday_daily_max']:,.0f}")
    print(f"Weekend vehicles/day     : {s['weekend_daily_min']:,.0f}-{s['weekend_daily_max']:,.0f}")
    print(f"Diurnal min / max        : {s['diurnal_min_hour']:02d}:00 ({s['diurnal_min_pct']:+.0f} %) / "
          f"{s['diurnal_max_hour']:02d}:00 ({s['diurnal_max_pct']:+.0f} %)")
    print(f"Mean tracked duration    : {s['mean_duration_sec']:.1f} s")
    print(f"Volume-speed Pearson r   : {s['volume_speed_pearson_r']:.2f} (3-h smoothed)")

    catalog.make_figure8(df, save_path=a.out)
    print(f"Figure -> {a.out}")


if __name__ == '__main__':
    main()
