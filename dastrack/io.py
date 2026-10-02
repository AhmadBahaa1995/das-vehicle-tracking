"""
HDF5 readers for DAS data stored as fixed-length segments in per-minute folders.

Expected archive layout::

    <data_root>/<YYYY>/<MM>/<DD>/<HH>/<MM>/*.h5

Each file holds an ``Acquisition/Raw[0]`` (or ``Acquisition``) group with a
``RawData`` dataset (time x channel or channel x time) and a ``RawDataTime``
dataset of microsecond UTC timestamps.

Two readers are provided:

* :func:`load_h5_folder_fast` - development / single-window reader.  Reads only
  the requested channel range, concatenates raw arrays and decimates by plain
  time slicing (much faster than per-trace ObsPy resampling).
* :func:`read_h5_stream` - the retry-guarded per-file reader used by the
  month-scale production run (:mod:`dastrack.production`).
"""
import gc
import logging
import os
import time

import h5py
import numpy as np
from obspy import Stream, Trace, UTCDateTime
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _raw_group(f):
    return f['Acquisition/Raw[0]'] if 'Acquisition/Raw[0]' in f else f['Acquisition']


def list_h5_files(folder, minute_start=0, minute_end=60):
    """Sorted list of .h5 files under ``folder/<MM>/`` for minutes in [start, end)."""
    paths = []
    for minute in range(minute_start, minute_end):
        sub = os.path.join(folder, f"{minute:02d}")
        if os.path.exists(sub):
            files = sorted(f for f in os.listdir(sub) if f.endswith('.h5'))
            paths.extend(os.path.join(sub, f) for f in files)
    return paths


def load_h5_folder_fast(file_paths, firstchannel=0, lastchannel=8250, target_fs=250,
                        verbose=True):
    """
    Fast DAS loader: concatenates raw numpy arrays across files, decimates by
    plain slicing (avoids ObsPy's slow per-trace resample), then wraps once in
    a Stream.  Returns an ObsPy Stream of channel traces.
    """
    if not file_paths:
        if verbose:
            print("No files found!")
        return Stream()

    data_chunks = []
    start_time_utc = None
    spatial_sampling = 2.0
    original_fs = 1000.0

    for path in tqdm(file_paths, desc="Reading H5 arrays", disable=not verbose):
        try:
            with h5py.File(path, 'r') as f:
                raw_group = _raw_group(f)
                dataset = raw_group['RawData']
                if start_time_utc is None:
                    dt_ds = raw_group['RawDataTime']
                    start_time_utc = UTCDateTime(np.copy(dt_ds[0]) / 1_000_000)
                    attrs = raw_group.attrs
                    parent_attrs = f['Acquisition'].attrs if 'Acquisition' in f else {}
                    spatial_sampling = float(attrs.get(
                        'SpatialSamplingInterval',
                        parent_attrs.get('SpatialSamplingInterval', 2.0)))
                    original_fs = float(attrs.get(
                        'PulseRate', attrs.get('AcquisitionFrequency', 1000.0)))
                raw_shape = dataset.shape
                time_axis = 0 if raw_shape[1] < raw_shape[0] else 1
                if time_axis == 0:
                    chunk = dataset[:, firstchannel:lastchannel].T   # (channels, time)
                else:
                    chunk = dataset[firstchannel:lastchannel, :]     # (channels, time)
                data_chunks.append(chunk)
        except Exception as e:
            print(f"  skipping {path}: {e}")

    if not data_chunks:
        return Stream()

    full_matrix = np.concatenate(data_chunks, axis=1)
    del data_chunks
    gc.collect()

    decimation_factor = int(original_fs // target_fs)
    if decimation_factor > 1:
        if verbose:
            print(f"Decimating {original_fs:.0f}Hz -> {target_fs}Hz (factor {decimation_factor})...")
        full_matrix = full_matrix[:, ::decimation_factor]

    traces = []
    for i in range(full_matrix.shape[0]):
        ch_num = firstchannel + i
        stats = {'network': 'DAS', 'station': f'{ch_num:04d}', 'channel': 'HSF',
                 'sampling_rate': target_fs, 'starttime': start_time_utc,
                 'distance': ch_num * spatial_sampling}
        traces.append(Trace(data=full_matrix[i], header=stats))
    return Stream(traces=traces)


def load_window(folder, first_channel, last_channel, minute_start=0, minute_end=10,
                target_fs=250, verbose=True):
    """
    Load a continuous window from one hour folder as a (time x channel) matrix.

    Returns ``(raw_matrix, fs, start_time_utc)`` or ``(None, None, None)`` if no
    files were found.
    """
    paths = list_h5_files(folder, minute_start, minute_end)
    if verbose:
        print(f"{len(paths)} .h5 files found under {folder}")
    stream = load_h5_folder_fast(paths, firstchannel=first_channel,
                                 lastchannel=last_channel, target_fs=target_fs,
                                 verbose=verbose)
    if len(stream) == 0:
        return None, None, None
    raw_matrix = np.stack([tr.data for tr in stream], axis=1)     # (time, channel)
    fs = float(stream[0].stats.sampling_rate)
    t0 = stream[0].stats.starttime
    return raw_matrix, fs, t0


def read_h5_stream(path, sampling_rate, channel_start, channel_end, max_retries=3):
    """
    Retry-guarded reader for one HDF5 segment (production run).

    Returns an ObsPy Stream with one trace per channel, resampled to
    ``sampling_rate`` if the file's native rate differs, or an empty Stream if
    the file cannot be read after ``max_retries`` attempts.
    """
    retry = 0
    while retry < max_retries:
        try:
            with h5py.File(path, 'r') as f:
                grp = _raw_group(f)
                dataset = grp['RawData']
                date_time = grp['RawDataTime']
                attrs = grp.attrs
                parent = f['Acquisition'].attrs if 'Acquisition' in f else {}
                spatial = float(attrs.get('SpatialSamplingInterval',
                                          parent.get('SpatialSamplingInterval', 2.0)))
                orig_fs = float(attrs.get('PulseRate', attrs.get('AcquisitionFrequency', 1000.0)))
                shape = dataset.shape
                time_axis = 0 if shape[1] < shape[0] else 1
                t_utc = UTCDateTime(np.copy(date_time[0]) / 1_000_000)
                d = dataset[:, channel_start:channel_end].T if time_axis == 0 \
                    else dataset[channel_start:channel_end, :]
            traces = []
            for x in range(d.shape[0]):
                ch = channel_start + x
                stats = {'network': 'DAS', 'station': f'{ch:04d}', 'location': '',
                         'channel': 'HSF', 'npts': len(d[x]),
                         'sampling_rate': orig_fs, 'starttime': t_utc,
                         'distance': ch * spatial}
                tr = Trace(data=d[x], header=stats)
                if orig_fs != sampling_rate:
                    tr.resample(sampling_rate)
                traces.append(tr)
            return Stream(traces=traces)
        except Exception as e:
            retry += 1
            if retry < max_retries:
                logger.warning(f"H5 read retry {retry}/{max_retries}: "
                               f"{os.path.basename(path)} - {str(e)[:80]}")
                time.sleep(0.5)
            else:
                logger.error(f"H5 read failed: {os.path.basename(path)}")
                return Stream()
