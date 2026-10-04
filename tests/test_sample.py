"""
Tests on the ten-minute sample from Zenodo (see data/README.md).

The sample-data tests are skipped if the sample has not been downloaded to
``data/sample``.
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

import dastrack as dt
from dastrack import production

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sample')
DATE, HOUR = '2024/02/02', 0
FIRST, LAST, MONITOR = 0, 500, 190
HOUR_DIR = os.path.join(ROOT, DATE, f'{HOUR:02d}')

needs_sample = pytest.mark.skipif(not os.path.isdir(HOUR_DIR),
                                  reason='sample data not downloaded (see data/README.md)')


def test_build_tasks():
    tasks = production.build_tasks('2024-02-01', '2024-02-28')
    assert len(tasks) == 672 and tasks[0] == ('2024/02/01', 0) and tasks[-1] == ('2024/02/28', 23)


@pytest.fixture(scope='module')
def window():
    raw, fs, _ = dt.load_window(HOUR_DIR, FIRST, LAST, 0, 10, target_fs=dt.FS, verbose=False)
    _, _, denoised = dt.preprocess(raw, fs, verbose=False)
    return denoised, fs


@needs_sample
def test_load_and_preprocess(window):
    denoised, fs = window
    assert fs == dt.FS
    assert denoised.shape[1] == LAST - FIRST
    assert denoised.shape[0] > 9 * 60 * fs            # ~10 min minus the 2 x 1 s edge trim
    assert denoised.min() >= 0 and denoised.max() <= 1


@needs_sample
def test_window_pipeline(window, tmp_path):
    denoised, fs = window
    starts, results, valid = dt.run_pipeline(denoised, fs, dt.DX, MONITOR,
                                             tracker_params=dt.TRACKER_DEVELOPMENT, verbose=False,
                                             plot_dir=str(tmp_path / 'figs'), **dt.DETECTION)
    assert len(starts) > 0 and len(valid) > 0
    for r in valid:
        assert dt.DETECTION['v_min_kmh'] <= r['speed_kmh'] <= dt.DETECTION['v_max_kmh']
        assert r['n_updates'] >= dt.TRACKER_DEVELOPMENT['min_track_updates']
        assert r['distance_m'] >= dt.TRACKER_DEVELOPMENT['min_track_dist_m']
        # sign convention: + means the track moves toward higher channel index
        assert np.sign(r['ch_exit'] - r['ch_entry']) == r['direction']
    assert os.path.exists(tmp_path / 'figs' / 'entry_detection.png')

    table = dt.results_table(results)
    assert len(table) == len([r for r in results if r]) and table['valid'].sum() == len(valid)


@needs_sample
def test_production_worker_and_catalog(tmp_path):
    rows = production.process_single_hour(
        (DATE, HOUR), data_root=ROOT, dx=dt.DX, fs_raw=dt.FS_RAW, fs_target=dt.FS,
        monitor_ch=MONITOR, ch_start=FIRST, ch_end=LAST, detection=dt.DETECTION,
        tracker_params=dt.TRACKER_PRODUCTION, plot_dir=str(tmp_path / 'qc'))
    assert len(rows) > 0
    assert os.path.exists(tmp_path / 'qc' / '2024-02-02' / 'Chunk_00-00.png')

    df = production.write_catalog(rows, str(tmp_path / 'catalog.csv'))
    assert list(df.columns) == production.CATALOG_COLUMNS
    assert (df['time'] >= pd.Timestamp('2024-02-02 09:00')).all()      # UTC -> JST
    assert (np.sign(df['speed']) == df['direction']).all()
