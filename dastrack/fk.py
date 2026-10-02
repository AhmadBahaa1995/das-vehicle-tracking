"""
F-K slant-stack velocity measurement (manuscript Eqs. 5-9).

Sign convention: a physical velocity ``v > 0`` means the vehicle moves toward
higher channel index.  With numpy's FFT convention such a streak maps onto the
line ``f = -v k`` in the F-K plane, i.e. its energy sits at (f > 0, k < 0).
"""
import numpy as np
from scipy.signal import find_peaks


def _fk_power(patch, fs, dx):
    nt, nx = patch.shape
    taper = np.outer(np.hanning(nt), np.hanning(nx))
    p2 = (patch - patch.mean()) * taper
    power = np.abs(np.fft.fftshift(np.fft.fft2(p2))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0 / fs))
    wvn = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    return power, freqs, wvn


def slant_stack(patch, fs, dx, v_max_mps=30.0, n=360, f_dc_exclude=0.08):
    """
    Slant-stack (phase-velocity) scan of a 2-D Hann-tapered space-time patch.

    Integrates F-K power along every candidate line through the origin,
    excluding ``|f| <= f_dc_exclude`` Hz.

    Returns
    -------
    v_phys : ndarray
        Candidate physical velocities (m/s), both directions.
    score : ndarray
        Mean F-K power along each line.
    fk_db, freqs, wvn : ndarray
        The F-K spectrum in dB and its axes (for plotting).
    All five are ``None`` if the patch is smaller than 8 x 8.
    """
    nt, nx = patch.shape
    if nt < 8 or nx < 8:
        return None, None, None, None, None
    power, freqs, wvn = _fk_power(patch, fs, dx)
    fk_db = 10 * np.log10(power + 1e-12)
    df = freqs[1] - freqs[0]
    f0 = freqs[0]
    vp = np.linspace(0.5, v_max_mps, n // 2)
    v_math = np.concatenate([-vp[::-1], vp])
    sc = np.zeros(len(v_math))
    kp = np.where(wvn > 1e-4)[0]
    kpv = wvn[kp]
    kn = np.where(wvn < -1e-4)[0]
    knv = wvn[kn]
    for i, v in enumerate(v_math):
        ki = kp if v > 0 else kn
        kv = kpv if v > 0 else knv
        fe = abs(v) * abs(kv)
        u = fe > f_dc_exclude
        if u.sum() == 0:
            continue
        fi = np.clip(np.round((fe[u] - f0) / df).astype(int), 0, len(freqs) - 1)
        sc[i] = power[fi, ki[u]].mean()
    return -v_math[::-1], sc[::-1], fk_db, freqs, wvn


def fk_candidates(v_phys, score, top_k=5, prom_frac=0.05):
    """
    Peaks of the slant-stack score, strongest first.

    Returns ``(candidates, contrast)`` where ``candidates`` is a list of
    ``(velocity_mps, power)`` and ``contrast`` = peak / median score.
    """
    if score is None or score.max() <= 0:
        return [], 1.0
    idx, _ = find_peaks(score, prominence=prom_frac * score.max())
    if len(idx) == 0:
        idx = np.array([int(np.argmax(score))])
    cands = sorted([(float(v_phys[i]), float(score[i])) for i in idx], key=lambda c: -c[1])
    contrast = float(score.max() / (np.median(score[score > 0]) + 1e-12))
    return cands[:top_k], contrast


def representative_ridge(patch, fs, dx, v_phys_ms, f_dc_exclude=0.08):
    """
    Strongest F-K point on the line ``f = -v k`` (f > 0) for plotting.

    Returns ``(f_hz, k_cyc_per_m)``.
    """
    power, freqs, wvn = _fk_power(patch, fs, dx)
    df = freqs[1] - freqs[0]
    f0 = freqs[0]
    v_math = -v_phys_ms
    ki = np.where(wvn > 1e-4)[0] if v_math > 0 else np.where(wvn < -1e-4)[0]
    kv = wvn[ki]
    fe = abs(v_math) * abs(kv)
    u = (fe > f_dc_exclude) & (fe < freqs[-1] * 0.9)
    if u.sum() == 0:
        return 0.0, 0.0
    fi = np.clip(np.round((fe[u] - f0) / df).astype(int), 0, len(freqs) - 1)
    bo = int(np.argmax(power[fi, ki[u]]))
    return float(fe[u][bo]), float(kv[u][bo])
