"""
Shared preprocessing chain (manuscript "Preprocessing" section, Eqs. 1-4).

bandpass (70-100 Hz, zero-phase SOS) -> |strain| -> mute bad channels ->
per-channel median subtraction -> robust 2-98 percentile normalisation ->
per-channel wavelet denoise (db4, 2 levels, hybrid hard/soft threshold).

All matrices are (time x channel).
"""
import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


def bandpass_matrix(raw_matrix, fs, fmin=70, fmax=100, order=4, trim_sec=1.0):
    """SOS band-pass along time (axis 0), then trim edge transients."""
    sos = butter(order, [fmin, fmax], btype='bandpass', fs=fs, output='sos')
    filt = sosfiltfilt(sos, raw_matrix, axis=0)
    trim = int(fs * trim_sec)
    if trim > 0 and filt.shape[0] > 2 * trim:
        filt = filt[trim:-trim, :]
    return filt


def custom_wavelet_threshold(coeffs, lam=0.15, a=0.5):
    """Hybrid hard/soft threshold (Eq. 4) applied to detail levels only."""
    out = []
    for i, c in enumerate(coeffs):
        if i == 0:
            out.append(c)
            continue
        c2 = np.copy(c)
        c2[c >= lam] = c[c >= lam] - a * lam
        c2[np.abs(c) < lam] = 0
        c2[c <= -lam] = c[c <= -lam] + a * lam
        out.append(c2)
    return out


def optimize_preprocessing(matrix_2d, bad_channels=((0, 0),),
                           wavelet='db4', level=2, lam=0.15, a=0.5, verbose=True):
    """
    |strain| -> mute bad channels -> per-channel median subtraction ->
    robust 2-98 percentile normalisation -> per-channel wavelet denoise.

    Parameters
    ----------
    matrix_2d : ndarray, (time, channel)
        Band-passed strain (or phase) data.
    bad_channels : sequence of (start, end)
        Channel ranges (relative to the slab) to mute; ``(0, 0)`` = none.

    Returns
    -------
    norm_data, denoised_matrix : ndarray, (time, channel), values in [0, 1]
    """
    log = print if verbose else (lambda *a, **k: None)

    log("  |strain| intensity...")
    data_intensity = np.abs(matrix_2d)

    log("  muting bad channels...")
    for s, e in bad_channels:
        if s < data_intensity.shape[1]:
            data_intensity[:, s:e] = 0

    log("  median subtraction...")
    channel_median = np.median(data_intensity, axis=0)
    data_no_static = np.clip(data_intensity - channel_median, 0, None)

    log("  percentile normalisation...")
    mask = np.ones(data_no_static.shape[1], dtype=bool)
    for s, e in bad_channels:
        if s < mask.size:
            mask[s:e] = False
    valid_cols = data_no_static[:, mask] if mask.any() else data_no_static
    p_min, p_max = np.percentile(valid_cols, [2, 98])
    if p_max - p_min < 1e-12:
        norm_data = data_no_static
    else:
        norm_data = np.clip((data_no_static - p_min) / (p_max - p_min), 0, 1)

    log("  wavelet denoising...")
    denoised = np.zeros_like(norm_data)
    for ch in tqdm(range(norm_data.shape[1]), desc="  wavelet ch", disable=not verbose):
        sig = norm_data[:, ch]
        if sig.max() == 0:
            continue
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        coeffs = custom_wavelet_threshold(coeffs, lam=lam, a=a)
        rec = pywt.waverec(coeffs, wavelet)
        denoised[:, ch] = rec[:len(sig)]
    return norm_data, np.clip(denoised, 0, 1)


def preprocess(raw_matrix, fs, bandpass=(70, 100), bad_channels=((0, 0),), verbose=True):
    """
    Full development chain: band-pass + edge trim, then :func:`optimize_preprocessing`.

    Returns ``(filtered_matrix, norm_data, denoised_matrix)``.
    """
    filtered = bandpass_matrix(raw_matrix, fs=fs, fmin=bandpass[0], fmax=bandpass[1],
                               order=4, trim_sec=1.0)
    norm_data, denoised = optimize_preprocessing(filtered, bad_channels=bad_channels,
                                                 verbose=verbose)
    return filtered, norm_data, denoised
