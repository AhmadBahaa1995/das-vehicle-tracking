"""
Month-scale production run (manuscript "Month-Scale Production Deployment").

A date range is split into hours; hours are distributed over worker processes;
each worker splits its hour into continuous 10-minute chunks (bounded memory),
runs load -> band-pass -> preprocess -> detect -> track on each chunk, and
returns one catalog row per valid vehicle.  Each chunk can also write a QC
image (waterfall with all tracks overlaid).
"""
import concurrent.futures
import gc
import logging
import os
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import pandas as pd
from obspy import Stream
from tqdm import tqdm

from .detection import detect_entries
from .io import list_h5_files, read_h5_stream
from .preprocessing import optimize_preprocessing
from .tracker import track_vehicle

logger = logging.getLogger(__name__)

CATALOG_COLUMNS = ['time', 'speed', 'direction', 'distance_m', 'duration_sec',
                   'speed_min_kmh', 'speed_max_kmh', 'n_updates',
                   'ch_entry', 'ch_exit', 'uncertainty_kmh']


def load_chunk(chunk_files, fs_raw, ch_start, ch_end, bandpass=(70, 100)):
    """
    Read and band-pass one chunk of consecutive HDF5 segments.

    Files whose channel count does not match ``ch_end - ch_start`` are skipped;
    gaps are interpolated when merging; 1 s is trimmed from the chunk start to
    remove the filter transient.  Returns ``(raw_matrix, chunk_start_utc)``
    with ``raw_matrix`` of shape (time, channel) at ``fs_raw``, or
    ``(None, None)`` if nothing could be read.
    """
    expected = ch_end - ch_start
    st = Stream()
    for fp in chunk_files:
        s = read_h5_stream(fp, sampling_rate=fs_raw, channel_start=ch_start, channel_end=ch_end)
        if len(s) == expected:
            st += s
    if len(st) == 0:
        return None, None
    st.merge(method=1, fill_value='interpolate')
    st.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])
    t0s = st[0].stats.starttime
    t1s = st[0].stats.endtime
    if (t1s - t0s) > 1.5:
        st.trim(t0s + 1, t1s)
    chunk_start = st[0].stats.starttime.datetime
    raw_matrix = np.array([tr.data for tr in st]).T
    return raw_matrix, chunk_start


def process_single_hour(task, data_root, dx, fs_raw, fs_target, monitor_ch,
                        ch_start, ch_end, detection, tracker_params,
                        files_per_chunk=20, plot_dir=None):
    """
    Worker: process one ``(date_str 'YYYY/MM/DD', hour)`` task.

    Returns a list of catalog rows (dicts), with ``time`` in UTC.
    """
    date_str, hour = task
    hour_folder = os.path.join(data_root, date_str, f"{hour:02d}")
    logger.info(f"[{date_str} {hour:02d}:00] worker start")
    if not os.path.exists(hour_folder):
        return []
    paths = list_h5_files(hour_folder, 0, 60)
    if not paths:
        return []

    chunks = [paths[i:i + files_per_chunk] for i in range(0, len(paths), files_per_chunk)]
    records = []
    for chunk_idx, chunk_files in enumerate(chunks):
        try:
            raw_matrix, chunk_start = load_chunk(chunk_files, fs_raw, ch_start, ch_end)
            if raw_matrix is None:
                continue

            # Decimate the band-passed matrix by plain time slicing, then denoise.
            decim = max(1, int(round(fs_raw / fs_target)))
            raw_matrix = raw_matrix[::decim, :]
            fs = fs_raw / decim

            _, denoised = optimize_preprocessing(raw_matrix, bad_channels=[(0, 0)], verbose=False)
            del raw_matrix
            gc.collect()

            starts, _ = detect_entries(denoised, fs=fs, monitor_channel=monitor_ch, monitor_width=5,
                                       cutoff_hz=1.5, min_height=detection['min_height'],
                                       min_distance_sec=detection['min_dist_sec'], verbose=False)

            results = []
            for t0 in starts:
                r = track_vehicle(t0, denoised, fs, dx, monitor_ch,
                                  v_min_kmh=detection['v_min_kmh'],
                                  v_max_kmh=detection['v_max_kmh'], **tracker_params)
                results.append(r)
                if r and r['valid']:
                    records.append({
                        'time': chunk_start + pd.Timedelta(seconds=r['t_sec']),
                        'speed': r['velocity_kmh'],
                        'direction': r['direction'],
                        'distance_m': r['distance_m'],
                        'duration_sec': r['duration_sec'],
                        'speed_min_kmh': r['speed_min_kmh'],
                        'speed_max_kmh': r['speed_max_kmh'],
                        'n_updates': r['n_updates'],
                        'ch_entry': r['ch_entry'], 'ch_exit': r['ch_exit'],
                        'uncertainty_kmh': r['kf_uncertainty_kmh'],
                    })

            if plot_dir:
                from .plotting import save_chunk_plot
                cur_min = chunk_idx * 10
                try:
                    save_chunk_plot(denoised, results, starts, monitor_ch, fs, dx,
                                    os.path.join(plot_dir, date_str.replace('/', '-'),
                                                 f"Chunk_{hour:02d}-{cur_min:02d}.png"),
                                    title_info=f"{date_str} {hour:02d}:{cur_min:02d} (UTC) | 10-min block")
                except Exception as e:
                    logger.error(f"[PLOT ERROR] {date_str} {hour:02d} chunk {chunk_idx}: {e}")
            del denoised
            gc.collect()
        except Exception as e:
            logger.error(f"[{date_str} {hour:02d}] chunk {chunk_idx} error: {str(e)[:150]}")
            continue
    return records


def build_tasks(start_date, end_date):
    """One ``('YYYY/MM/DD', hour)`` task per hour, ``end_date`` inclusive."""
    tasks = []
    cur = datetime.strptime(start_date, '%Y-%m-%d')
    d1 = datetime.strptime(end_date, '%Y-%m-%d')
    while cur <= d1:
        ds = cur.strftime('%Y/%m/%d')
        tasks.extend((ds, hour) for hour in range(24))
        cur += timedelta(days=1)
    return tasks


def run_production(start_date, end_date, max_workers=8, **worker_kwargs):
    """Run :func:`process_single_hour` over every hour in parallel; return all rows."""
    tasks = build_tasks(start_date, end_date)
    worker = partial(process_single_hour, **worker_kwargs)
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Progress", unit="hr") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                try:
                    records.extend(fut.result())
                except Exception as e:
                    logger.error(f"hour {futures[fut]} error: {e}")
                finally:
                    pbar.update(1)
    return records


def write_catalog(records, csv_path, utc_offset_hours=9):
    """
    Write the vehicle catalog, one row per vehicle, sorted by time.

    ``time`` is converted from UTC to local time by adding ``utc_offset_hours``
    (9 = JST).  ``speed`` is signed (+ toward higher channel index).
    """
    df = pd.DataFrame(records, columns=CATALOG_COLUMNS)
    df['time'] = pd.to_datetime(df['time']) + pd.Timedelta(hours=utc_offset_hours)
    df = df.sort_values('time').reset_index(drop=True)
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
