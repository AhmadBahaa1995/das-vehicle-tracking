"""
HDF5 readers for DAS data.

The full archive is stored as fixed-length segments in per-minute folders::

    <data_root>/<YYYY>/<MM>/<DD>/<HH>/<MM>/*.h5

Each file holds an ``Acquisition/Raw[0]`` (or ``Acquisition``) group with a
``RawData`` dataset (time x channel or channel x time) and a ``RawDataTime``
dataset of UTC timestamps.

Files without that structure or without metadata attributes (such as the
public sample) are also read: the raw data is taken to be the largest 2-D
dataset and the timestamps a 1-D dataset matching its time axis.  Missing
attributes are handled as follows:

* sampling rate - inferred from the per-sample timestamps (else 1000 Hz),
* channel spacing - not needed by the pipeline, which uses ``params.DX``,
* timestamp unit - inferred from the magnitude (s, ms, us or ns since epoch).

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


def _epoch_seconds(values):
    """Timestamps since the Unix epoch in s, ms, us or ns -> float seconds."""
    v = np.asarray(values, dtype=np.float64)
    ref = abs(float(v.flat[0])) if v.size else 0.0
    scale = 1e9 if ref > 1e17 else 1e6 if ref > 1e14 else 1e3 if ref > 1e11 else 1.0
    return v / scale


def _start_time(time_ds):
    t0 = time_ds[0]
    if isinstance(t0, (bytes, str, np.bytes_, np.str_)):
        return UTCDateTime(t0.decode() if isinstance(t0, bytes) else str(t0))
    return UTCDateTime(float(_epoch_seconds(np.copy(t0))))


def _find_datasets(f):
    """
    Return ``(data_ds, time_ds, attrs, parent_attrs)`` for one open file.

    Uses the archive structure when present, otherwise the largest 2-D dataset
    and the 1-D dataset whose length matches its time axis.
    """
    for path in ('Acquisition/Raw[0]', 'Acquisition'):
        if path in f and isinstance(f[path], h5py.Group) and 'RawData' in f[path]:
            grp = f[path]
            parent = f['Acquisition'].attrs if 'Acquisition' in f else {}
            return grp['RawData'], grp.get('RawDataTime'), grp.attrs, parent

    found = []
    f.visititems(lambda name, obj: found.append(obj) if isinstance(obj, h5py.Dataset) else None)
    two_d = [d for d in found if d.ndim == 2]
    if not two_d:
        raise ValueError("no 2-D data array found")
    data = max(two_d, key=lambda d: d.size)
    n_time = max(data.shape)
    one_d = [d for d in found if d.ndim == 1 and d.shape[0] in data.shape]
    one_d.sort(key=lambda d: ('time' not in d.name.lower(), d.shape[0] != n_time))
    time_ds = one_d[0] if one_d else None
    return data, time_ds, data.attrs, data.parent.attrs


def _file_info(f):
    """
    Data dataset, time axis, start time, native sampling rate and channel spacing
    of one open file.  Attribute lookups follow the archive conventions; the
    sampling rate falls back to the timestamps, then to 1000 Hz.
    """
    dataset, time_ds, attrs, parent_attrs = _find_datasets(f)
    shape = dataset.shape
    time_axis = 0 if shape[1] < shape[0] else 1
    spatial = float(attrs.get('SpatialSamplingInterval',
                              parent_attrs.get('SpatialSamplingInterval', 2.0)))
    fs = attrs.get('PulseRate', attrs.get('AcquisitionFrequency', None))
    if fs is None:
        fs = 1000.0
        if time_ds is not None and time_ds.shape[0] == shape[time_axis] and time_ds.dtype.kind in 'iuf':
            dt = np.median(np.diff(_epoch_seconds(time_ds[:2001])))
            if dt > 0:
                fs = float(round(1.0 / dt))
    t0 = _start_time(time_ds) if time_ds is not None else UTCDateTime(0)
    return dataset, time_axis, t0, float(fs), spatial


def list_input_files(path):
    """A single .h5 file, or the sorted .h5/.hdf5 files directly inside a folder."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path))
                if f.lower().endswith(('.h5', '.hdf5'))]
    return []


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
                dataset, time_axis, t0, fs_file, spatial_file = _file_info(f)
                if start_time_utc is None:
                    start_time_utc, original_fs, spatial_sampling = t0, fs_file, spatial_file
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
    Load a continuous window from one hour folder of the archive as a
    (time x channel) matrix.

    Returns ``(raw_matrix, fs, start_time_utc)`` or ``(None, None, None)`` if no
    files were found.
    """
    paths = list_h5_files(folder, minute_start, minute_end)
    if verbose:
        print(f"{len(paths)} .h5 files found under {folder}")
    return load_files(paths, first_channel, last_channel, target_fs, verbose)


def load_input(path, first_channel, last_channel, target_fs=250, verbose=True):
    """
    Load a single .h5 file, or every .h5 file directly inside a folder (in name
    order, concatenated in time), as a (time x channel) matrix.

    Returns ``(raw_matrix, fs, start_time_utc)`` or ``(None, None, None)``.
    """
    paths = list_input_files(path)
    if verbose:
        print(f"{len(paths)} .h5 file(s) found at {path}")
    return load_files(paths, first_channel, last_channel, target_fs, verbose)


def load_files(paths, first_channel, last_channel, target_fs=250, verbose=True):
    """Load and concatenate the given files; see :func:`load_window`."""
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
                dataset, time_axis, t_utc, orig_fs, spatial = _file_info(f)
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
