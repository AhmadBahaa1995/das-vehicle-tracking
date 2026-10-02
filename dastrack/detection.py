"""
Vehicle detection at a monitor channel (manuscript "Vehicle detection").

A narrow band of channels at the monitor position is averaged, smoothed with a
low-order Butterworth low-pass, and a derivative-based peak search returns the
crossing times above ``min_height`` separated by at least ``min_distance_sec``.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def lowpass_envelope(signal, fs, cutoff_hz=1.5, order=2):
    sos = butter(order, cutoff_hz, btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, signal.astype(np.float64))


def peaks_location_search(V, min_height=0.2, min_distance_samples=10):
    """Local maxima of ``V`` (sign-of-derivative search) above ``min_height``."""
    V = np.array(V, dtype=float)
    if len(V) < 3:
        return []
    S = np.sign(np.diff(V)).astype(int)
    for i in range(len(S) - 2, -1, -1):
        if S[i] == 0:
            S[i] = 1 if S[i + 1] >= 0 else -1
    R = np.diff(S.astype(float))
    raw = [i + 1 for i in range(len(R)) if R[i] == -2 and V[i + 1] >= min_height]
    valid, last = [], -min_distance_samples
    for p in raw:
        if p - last >= min_distance_samples:
            valid.append(p)
            last = p
    return valid


def detect_entries(clean_matrix, fs, monitor_channel=60, monitor_width=5,
                   cutoff_hz=1.5, min_height=0.25, min_distance_sec=10.0, verbose=True):
    """
    Detect vehicle crossings at ``monitor_channel`` (relative to the slab).

    Returns ``(start_samples, smoothed_envelope)``.
    """
    ch_lo = max(0, monitor_channel)
    ch_hi = min(clean_matrix.shape[1], monitor_channel + monitor_width)
    sig = np.mean(clean_matrix[:, ch_lo:ch_hi], axis=1)
    sm = lowpass_envelope(sig, fs, cutoff_hz=cutoff_hz)
    starts = peaks_location_search(sm, min_height=min_height,
                                   min_distance_samples=int(min_distance_sec * fs))
    if verbose:
        print(f"\n{'=' * 60}\nENTRY DETECTION")
        print(f"  Monitor ch {monitor_channel} | thresh {min_height} | min gap {min_distance_sec}s")
        print(f"  -> {len(starts)} vehicle(s) at t(s): {[round(s / fs, 1) for s in starts]}")
        print(f"{'=' * 60}")
    return starts, sm
