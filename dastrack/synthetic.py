"""
Synthetic DAS archive for trying the pipeline without the (proprietary) data.

Writes HDF5 segments in the layout the readers expect::

    <root>/<YYYY>/<MM>/<DD>/<HH>/<MM>/<name>.h5
        Acquisition/Raw[0]/RawData      (time x channel, float32)
        Acquisition/Raw[0]/RawDataTime  (microseconds since epoch, int64)

Each vehicle is a moving Gaussian envelope (along the fibre) modulating a
70-100 Hz carrier, on top of broadband noise.
"""
import os
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np


def random_vehicles(duration_sec, monitor_channel, n_vehicles, seed=0,
                    speed_range_kmh=(20, 45)):
    """List of dicts ``(t_cross_sec, v_mps, amp, f_hz)``; each crosses ``monitor_channel`` at ``t_cross_sec``."""
    rng = np.random.default_rng(seed)
    t_cross = np.sort(rng.uniform(10, duration_sec - 10, n_vehicles))
    return [dict(t_cross_sec=float(tc),
                 v_mps=float(rng.choice([-1, 1]) * rng.uniform(*speed_range_kmh) / 3.6),
                 amp=float(rng.uniform(0.6, 1.0)),
                 f_hz=float(rng.uniform(75, 95)))
            for tc in t_cross]


def synthetic_strain(t, n_channels, vehicles, monitor_channel, dx, width_ch=2.5,
                     noise=0.3, seed=0):
    """Raw strain matrix (len(t) x n_channels) for the given vehicles."""
    rng = np.random.default_rng(seed)
    x = np.arange(n_channels)
    data = noise * rng.standard_normal((len(t), n_channels)).astype(np.float32)
    for v in vehicles:
        c = monitor_channel + v['v_mps'] * (t - v['t_cross_sec']) / dx
        near = np.abs(c - n_channels / 2) < n_channels / 2 + 10 * width_ch
        if not near.any():
            continue
        env = np.exp(-0.5 * ((x[None, :] - c[near, None]) / width_ch) ** 2)
        carrier = np.sin(2 * np.pi * v['f_hz'] * t[near] + rng.uniform(0, 2 * np.pi))
        data[near] += (v['amp'] * env * carrier[:, None]).astype(np.float32)
    return data


def make_synthetic_archive(root, start_utc='2024-02-02 00:00:00', minutes=10,
                           n_channels=300, monitor_channel=150, dx=2.0419, fs=1000.0,
                           segment_sec=30.0, vehicles_per_minute=4, seed=0):
    """
    Write ``minutes`` minutes of synthetic data starting at ``start_utc``.

    Returns the list of vehicles (``t_cross_sec`` relative to ``start_utc``).
    """
    t_start = datetime.strptime(start_utc, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    duration = minutes * 60.0
    vehicles = random_vehicles(duration, monitor_channel, int(vehicles_per_minute * minutes), seed=seed)
    n_seg = int(round(duration / segment_sec))
    ns = int(segment_sec * fs)
    for k in range(n_seg):
        seg_t0 = k * segment_sec
        t = seg_t0 + np.arange(ns) / fs
        data = synthetic_strain(t, n_channels, vehicles, monitor_channel, dx, seed=seed + k + 1)
        wall = t_start + timedelta(seconds=seg_t0)
        folder = os.path.join(root, wall.strftime('%Y/%m/%d/%H/%M'))
        os.makedirs(folder, exist_ok=True)
        with h5py.File(os.path.join(folder, wall.strftime('synthetic_%Y%m%dT%H%M%SZ.h5')), 'w') as f:
            acq = f.create_group('Acquisition')
            acq.attrs['SpatialSamplingInterval'] = dx
            raw = acq.create_group('Raw[0]')
            raw.attrs['AcquisitionFrequency'] = fs
            raw.create_dataset('RawData', data=data)
            us0 = int(wall.timestamp() * 1_000_000)
            raw.create_dataset('RawDataTime', data=us0 + (np.arange(ns) * 1e6 / fs).astype(np.int64))
    return vehicles
