"""End-to-end checks on a small synthetic archive."""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

import dastrack as dt
from dastrack import production
from dastrack.synthetic import make_synthetic_archive

FIRST, LAST, MONITOR_ABS = 20, 220, 110
MONITOR_REL = MONITOR_ABS - FIRST


@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('archive'))
    vehicles = make_synthetic_archive(root, minutes=3, n_channels=240, monitor_channel=MONITOR_ABS,
                                      vehicles_per_minute=3, seed=7)
    return root, vehicles


def _match(truth, t_sec, tol=1.5):
    hits = [v for v in truth if abs(v['t_cross_sec'] - t_sec) < tol]
    return hits[0] if hits else None


def test_window_pipeline_recovers_vehicles(archive, tmp_path):
    root, truth = archive
    raw, fs, _ = dt.load_window(os.path.join(root, '2024/02/02/00'), FIRST, LAST, 0, 3,
                                target_fs=dt.FS, verbose=False)
    assert raw.shape[1] == LAST - FIRST and fs == dt.FS
    _, _, denoised = dt.preprocess(raw, fs, verbose=False)
    assert denoised.min() >= 0 and denoised.max() <= 1

    _, results, valid = dt.run_pipeline(denoised, fs, dt.DX, MONITOR_REL,
                                        tracker_params=dt.TRACKER_DEVELOPMENT, verbose=False,
                                        plot_dir=str(tmp_path / 'figs'), **dt.DETECTION)
    assert len(valid) >= 0.6 * len(truth)
    trim = 1.0     # band-pass edge trim shifts the time origin by 1 s
    n_matched = 0
    for r in valid:
        v = _match(truth, r['t_sec'] + trim)
        if v is None:
            continue
        n_matched += 1
        assert np.sign(r['velocity_kmh']) == np.sign(v['v_mps'])
        assert abs(r['speed_kmh'] - abs(v['v_mps']) * 3.6) < 8.0
        assert r['distance_m'] >= dt.TRACKER_DEVELOPMENT['min_track_dist_m']
    assert n_matched >= 0.8 * len(valid)
    assert os.path.exists(tmp_path / 'figs' / 'vehicle_001.png')

    table = dt.results_table(results)
    assert len(table) == len(results) and table['valid'].sum() == len(valid)


def test_production_hour_and_catalog(archive, tmp_path):
    root, truth = archive
    rows = production.process_single_hour(
        ('2024/02/02', 0), data_root=root, dx=dt.DX, fs_raw=dt.FS_RAW, fs_target=dt.FS,
        monitor_ch=MONITOR_REL, ch_start=FIRST, ch_end=LAST, detection=dt.DETECTION,
        tracker_params=dt.TRACKER_PRODUCTION, plot_dir=str(tmp_path / 'qc'))
    assert len(rows) >= 0.6 * len(truth)
    assert os.path.exists(tmp_path / 'qc' / '2024-02-02' / 'Chunk_00-00.png')

    df = production.write_catalog(rows, str(tmp_path / 'catalog.csv'))
    assert list(df.columns) == production.CATALOG_COLUMNS
    assert (df['time'] >= pd.Timestamp('2024-02-02 09:00')).all()      # UTC -> JST
    assert set(df['direction']) <= {-1, 1}
    assert (np.sign(df['speed']) == df['direction']).all()


def test_build_tasks():
    tasks = production.build_tasks('2024-02-01', '2024-02-28')
    assert len(tasks) == 672 and tasks[0] == ('2024/02/01', 0) and tasks[-1] == ('2024/02/28', 23)
