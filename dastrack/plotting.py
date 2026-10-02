"""
Figures: preprocessing chain, entry detection, per-vehicle diagnostic
(manuscript Figures 5 and 9) and the per-chunk quality-control image written
by the production run.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from .fk import representative_ridge

PAPER_RC = {
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.edgecolor': 'black',
    'axes.labelcolor': 'black', 'text.color': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.grid': False, 'font.size': 10.5,
    'legend.frameon': True, 'legend.facecolor': 'white',
    'legend.edgecolor': '0.65', 'legend.framealpha': 0.99,
    'axes.titleweight': 'bold',
}
TRACK_C, ENTRY_C, EXIT_C, DET_C = '#15803d', '#0891b2', '#a21caf', '#dc2626'
RANK_C = ['#dc2626', '#f59e0b', '#6b7280']   # peak #1 / #2 / #3+


def out_path(directory, name):
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def _finish(fig, save_path, show, dpi=200):
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_preprocessing(filtered_matrix, norm_data, denoised_matrix, fs=None,
                       save_path=None, show=True):
    """Three-panel view of the preprocessing chain."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        m, n = denoised_matrix.shape
        ext = [0, n, m / fs, 0] if fs else None
        ylab = "Time (s)" if fs else "Time (samples)"
        vmn, vmx = np.percentile(filtered_matrix, [2, 98])
        axes[0].imshow(filtered_matrix, aspect='auto', cmap='seismic', vmin=vmn, vmax=vmx, extent=ext)
        axes[0].set_title("Band-pass filtered")
        axes[0].set_ylabel(ylab)
        axes[1].imshow(norm_data, aspect='auto', cmap='magma', vmin=0, vmax=1, extent=ext)
        axes[1].set_title("|.| + median-subtracted + normalised")
        axes[2].imshow(denoised_matrix, aspect='auto', cmap='magma', vmin=0, vmax=1, extent=ext)
        axes[2].set_title("Wavelet denoised")
        for ax in axes:
            ax.set_xlabel("Channel")
        plt.tight_layout()
        _finish(fig, save_path, show)


def plot_entry_detection(clean_matrix, smoothed, start_times, monitor_channel, min_height, fs,
                         save_path=None, show=True):
    """Monitor envelope with detections, and the waterfall with detection times."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        t = np.arange(len(smoothed)) / fs
        ch_hi = min(clean_matrix.shape[1], monitor_channel + 5)
        axes[0].plot(t, np.mean(clean_matrix[:, monitor_channel:ch_hi], axis=1),
                     color='lightsteelblue', alpha=0.8, label='Monitor signal')
        axes[0].plot(t, smoothed, color='navy', lw=1.8, label='Low-pass envelope')
        if start_times:
            axes[0].plot(np.array(start_times) / fs, smoothed[start_times], 'o', color=DET_C,
                         ms=7, label=f'Detections ({len(start_times)})')
        axes[0].axhline(min_height, color='gray', ls='--', label='Threshold')
        axes[0].set_title(f'Entry detection (monitor ch {monitor_channel})')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend(fontsize=8)
        m = clean_matrix.shape[0]
        axes[1].imshow(clean_matrix, aspect='auto', cmap='magma', vmin=0, vmax=1,
                       extent=[0, clean_matrix.shape[1], m / fs, 0])
        for p in start_times:
            axes[1].axhline(p / fs, color=DET_C, ls='--', lw=0.8, alpha=0.6)
        axes[1].axvline(monitor_channel, color='cyan', ls=':', lw=2, label='Monitor')
        axes[1].set_title('Waterfall (dashed = detection times)')
        axes[1].set_xlabel('Channel')
        axes[1].set_ylabel('Time (s)')
        axes[1].legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        _finish(fig, save_path, show)


def plot_vehicle_diagnostic(result, matrix, fs, dx, background_tracks=(), title=None,
                            save_path=None, show=True):
    """
    Six-panel diagnostic for one tracked vehicle (manuscript Figures 5 and 9):
    (A) lifecycle on the full waterfall, (B) track zoom, (C) bootstrap F-K
    spectrum with the winning line, (D) slant-stack peaks and the chosen seed,
    (E) RTS speed profile with uncertainty, (F) summary.

    ``background_tracks`` is an optional list of ``(t_sec, ch)`` arrays drawn
    faded in panels A/B (other vehicles in the same window).
    """
    d = result['diag']
    t_ord = np.asarray(result['profile_t_sec'])
    sm_ch = np.asarray(result['track_ch'])
    sm_v = np.asarray(result['speed_profile_kmh'])
    sm_sig = np.asarray(result['speed_sigma_kmh'])
    t_det = result['t_sec']
    ch_det = result['ch_det']
    v_seed = result['seed_kmh'] / 3.6
    final_kmh = result['velocity_kmh']
    t_entry, t_exit = result['t_entry_sec'], result['t_exit_sec']
    ch_entry, ch_exit = result['ch_entry'], result['ch_exit']
    dist_m = result['distance_m']
    direction = '→ (+)' if result['direction'] > 0 else '← (−)'
    raw_ord, rank_ord, all_cands = d['raw_kmh'], d['rank'], d['all_cands']
    meas = [(t_ord[i], abs(raw_ord[i]), rank_ord[i]) for i in range(len(t_ord))
            if raw_ord[i] is not None]
    meas_t = np.array([x[0] for x in meas])
    meas_v = np.array([x[1] for x in meas])
    ranks = np.array([x[2] for x in meas])

    with plt.rc_context(PAPER_RC):
        m, n = matrix.shape
        fig, axes = plt.subplots(2, 3, figsize=(21, 10))
        fig.suptitle(title or
                     f"Vehicle {result['car_id']} — detected @ {t_det:.2f} s | {abs(final_kmh):.1f} km/h "
                     f"| on road {t_entry:.0f}–{t_exit:.0f} s, ch {ch_entry:.0f}–{ch_exit:.0f} "
                     f"({dist_m:.0f} m) | {direction}",
                     fontsize=12.5, fontweight='bold')

        # A: full waterfall + lifecycle
        ax = axes[0, 0]
        ax.imshow(matrix, aspect='auto', cmap='magma', vmin=0, vmax=1, extent=[0, n, m / fs, 0])
        for bt, bch in background_tracks:
            ax.plot(bch, bt, '--', color='#70b396', lw=0.6, alpha=0.6, zorder=4)
        ax.plot(sm_ch, t_ord, color=TRACK_C, lw=2.5, label='RTS track', zorder=5)
        ax.plot(ch_det, t_det, '*', color=DET_C, ms=13, mec='white', mew=0.6, zorder=6,
                label='Detection')
        ax.plot(ch_entry, t_entry, 'o', color=ENTRY_C, ms=9, mec='white', mew=0.6, zorder=6,
                label=f'Entry t{t_entry:.0f}s ch{ch_entry:.0f}')
        ax.plot(ch_exit, t_exit, 's', color=EXIT_C, ms=9, mec='white', mew=0.6, zorder=6,
                label=f'Exit t{t_exit:.0f}s ch{ch_exit:.0f}')
        ax.axvline(d['monitor_channel'], color='#eab308', ls=':', lw=1.2, alpha=0.8)
        ax.set_title('A — Full waterfall + lifecycle')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Time (s)')
        ax.legend(fontsize=8, loc='upper right')

        # B: track zoom
        ax = axes[0, 1]
        ch_lo = int(max(0, np.min(sm_ch) - 25))
        ch_hi = int(min(n - 1, np.max(sm_ch) + 25))
        tt_lo = max(0, int((t_ord.min() - 2) * fs))
        tt_hi = min(m - 1, int((t_ord.max() + 2) * fs))
        ax.imshow(matrix[tt_lo:tt_hi + 1, ch_lo:ch_hi + 1], aspect='auto', cmap='magma',
                  vmin=0, vmax=1, extent=[ch_lo, ch_hi, tt_hi / fs, tt_lo / fs])
        for bt, bch in background_tracks:
            ax.plot(bch, bt, '--', color='#70b396', lw=2.0, alpha=0.6, zorder=4)
        if len(background_tracks):
            ax.plot([], [], '--', color='#70b396', lw=1.3, alpha=0.6, label='other tracked vehicles')
        ax.plot(sm_ch, t_ord, color=TRACK_C, lw=2.5, label='RTS track', zorder=5)
        ax.scatter(sm_ch, t_ord, c=TRACK_C, s=24, edgecolors='white', lw=0.6, zorder=5,
                   label='KF states')
        ax.plot(ch_det, t_det, '*', color=DET_C, ms=15, mec='white', mew=0.6, zorder=6)
        ax.set_xlim(ch_lo, ch_hi)
        ax.set_ylim(tt_hi / fs, tt_lo / fs)
        ax.set_title('B — Track zoom')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Time (s)')
        ax.legend(fontsize=8, loc='upper right')

        # C: bootstrap F-K spectrum.  A vehicle at velocity v (v > 0 = toward
        # higher channel) lies on f = -v k; the image is drawn with +f upward.
        ax = axes[1, 0]
        fk_db0, freqs0, wvn0 = d['fk_db0'], d['freqs0'], d['wvn0']
        pf, pk = representative_ridge(d['patch0'], fs, dx, v_seed)
        vlo, vhi = np.percentile(fk_db0, [35, 99.5])
        im = ax.imshow(fk_db0, aspect='auto', cmap='jet', origin='lower', vmin=vlo, vmax=vhi,
                       extent=[wvn0[0], wvn0[-1], freqs0[0], freqs0[-1]])
        ax.axhspan(-0.08, 0.08, color='white', alpha=0.2, lw=0, zorder=1)
        ax.axvline(0, color='white', ls='--', lw=0.8, alpha=0.5)
        ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)
        k_ln = np.linspace(wvn0[0], wvn0[-1], 300)
        ax.plot(k_ln, -v_seed * k_ln, 'w--', lw=2, alpha=0.9, label=f'f = −vk, v = {v_seed:+.2f} m/s')
        ax.plot(pk, pf, '*', color='white', mec='black', mew=0.6, ms=12,
                label=f'Ridge maximum ({pf:.3f} Hz)')
        ax.set_ylim(-10, 10)
        ax.set_title('C — Bootstrap F-K spectrum')
        ax.set_xlabel('Wavenumber k (cycles/m)')
        ax.set_ylabel('Frequency f (Hz)')
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, label='Power (dB)')

        # D: slant-stack score, ranked peaks, chosen seed
        ax = axes[1, 1]
        if d['v_ax0'] is not None:
            ax.plot(d['v_ax0'] * 3.6, d['sc0'], color='#1f77b4', lw=1.4, label='Slant-stack score')
            for rank, (v, p) in enumerate(all_cands[:5]):
                color = RANK_C[min(rank, 2)]
                ax.plot(v * 3.6, p, 'o', color=color, ms=9, zorder=5)
                ax.annotate(f'#{rank + 1}\n{v * 3.6:+.0f}', (v * 3.6, p), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color=color)
            ax.axvline(v_seed * 3.6, color=TRACK_C, lw=2, ls='--',
                       label=f'Chosen seed {v_seed * 3.6:+.0f} (coherence score)')
            ax.axvline(final_kmh, color=DET_C, lw=1.3, ls=':', label=f'Final {final_kmh:+.0f}')
        ax.set_title('D — Slant-stack peaks')
        ax.set_xlabel('Velocity (km/h)')
        ax.set_ylabel('Mean F-K power on line')
        ax.legend(fontsize=8)

        # E: speed profile (measurements coloured by the peak rank used)
        ax = axes[0, 2]
        for r, lbl in [(0, 'peak #1'), (1, 'peak #2'), (2, 'peak #3+')]:
            sel = (ranks == r) if r < 2 else (ranks >= 2)
            if np.any(sel):
                ax.plot(meas_t[sel], meas_v[sel], 'o', color=RANK_C[r], ms=6, alpha=0.9,
                        label=f'F-K measurement ({lbl})')
        ax.plot(t_ord, np.abs(sm_v), '-', color=TRACK_C, lw=2.5, label='RTS profile')
        ax.fill_between(t_ord, np.abs(sm_v) - sm_sig, np.abs(sm_v) + sm_sig, alpha=0.15, color=TRACK_C)
        ax.axvline(t_det, color=DET_C, ls='--', lw=1.5, label=f'At monitor {abs(final_kmh):.1f}')
        ax.axvline(t_entry, color=ENTRY_C, lw=1, alpha=0.8)
        ax.axvline(t_exit, color=EXIT_C, lw=1, alpha=0.8)
        for lim in (d['v_min_kmh'], d['v_max_kmh']):
            ax.axhline(lim, color='#9ca3af', ls=':', lw=0.8)
        ax.set_title('E — Speed profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.legend(fontsize=8)

        # F: summary
        ax = axes[1, 2]
        ax.axis('off')
        n2 = int(np.sum(ranks >= 1)) if len(ranks) else 0
        ax.text(0.05, 0.5,
                f"Vehicle {result['car_id']}\n\n"
                f"Detected       : {t_det:.2f} s\n"
                f"Bootstrap #1   : {result['bootstrap_kmh']:+.1f} km/h\n"
                f"Seed chosen    : {result['seed_kmh']:+.1f} km/h\n\n"
                f"Speed @ monitor: {abs(final_kmh):.1f} ± {result['kf_uncertainty_kmh']:.1f} km/h\n"
                f"Profile range  : {np.abs(sm_v).min():.1f}–{np.abs(sm_v).max():.1f} km/h\n"
                f"Direction      : {direction}\n\n"
                f"Entry : t={t_entry:.1f}s  ch={ch_entry:.0f}\n"
                f"Exit  : t={t_exit:.1f}s  ch={ch_exit:.0f}\n"
                f"On road: {t_exit - t_entry:.1f} s, {dist_m:.0f} m\n"
                f"Updates: {len(meas_t)} ({n2} used 2nd+ peak)\n\n"
                f"Valid          : {'YES' if result['valid'] else 'NO'}",
                transform=ax.transAxes, fontsize=10.5, va='center', fontfamily='monospace',
                bbox=dict(facecolor='white', edgecolor='#334155', lw=1.1, pad=12))

        plt.tight_layout()
        _finish(fig, save_path, show)


def track_from_result(result):
    """(t_sec, ch) of a result's RTS track, for use as a background track."""
    if not result or len(result['profile_t_sec']) < 2:
        return None
    return np.asarray(result['profile_t_sec']), np.asarray(result['track_ch'])


# ─────────────────────────────────────────────────────────────────────────────
# Production QC image
# ─────────────────────────────────────────────────────────────────────────────

def _predicted_extension(result, fs, dx, n_ch, n_time_samples, extend_sec=8.0):
    """Constant-velocity extrapolation before entry / after exit (dashed lines)."""
    if result is None or len(result['profile_t_sec']) < 2:
        return None, None
    t = np.array(result['profile_t_sec'])
    ch = np.array(result['track_ch'])
    v = np.array(result['speed_profile_kmh'])
    v0 = v[0] / 3.6
    vN = v[-1] / 3.6
    npts = int(extend_sec * fs)
    t_pre = np.arange(t[0] * fs - npts, t[0] * fs)
    ch_pre = ch[0] + (t_pre - t[0] * fs) * (v0 / fs) / dx
    t_post = np.arange(t[-1] * fs + 1, t[-1] * fs + npts)
    ch_post = ch[-1] + (t_post - t[-1] * fs) * (vN / fs) / dx

    def _clip(ti, ci):
        keep = (ti >= 0) & (ti < n_time_samples) & (ci >= 0) & (ci < n_ch)
        return ci[keep], ti[keep]
    return _clip(t_pre, ch_pre), _clip(t_post, ch_post)


def save_chunk_plot(matrix, results, start_times, monitor_ch, fs, dx, filepath, title_info=""):
    """Waterfall of one 10-minute chunk with every valid RTS track overlaid."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    m, n = matrix.shape
    fig, ax = plt.subplots(figsize=(24, 12))
    ax.imshow(matrix, aspect='auto', cmap='magma', vmin=0, vmax=1)
    if monitor_ch is not None:
        ax.axvline(monitor_ch, color='cyan', ls=':', lw=2.5, label='Monitor channel', alpha=0.9)

    valid = [r for r in results if r and r['valid']]
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(valid))))
    for idx, (r, col) in enumerate(zip(valid, colors)):
        t_smp = np.array(r['profile_t_sec']) * fs
        ch_arr = np.array(r['track_ch'])
        ax.plot(ch_arr, t_smp, color=col, lw=2.0, alpha=0.9)
        for seg in _predicted_extension(r, fs, dx, n, m):
            if seg is not None and len(seg[0]) > 1:
                ax.plot(seg[0], seg[1], color=col, lw=1.4, ls='--', alpha=0.55)
        ax.plot(r['ch_entry'], r['t_entry_sec'] * fs, 'o', color='cyan', ms=6, zorder=5)
        ax.plot(r['ch_exit'], r['t_exit_sec'] * fs, 's', color='magenta', ms=6, zorder=5)
        mid = len(ch_arr) // 2
        sym = '←' if r['direction'] < 0 else '→'
        ax.text(ch_arr[mid] + 4, t_smp[mid], f"V{idx + 1}:{r['speed_kmh']:.0f}{sym}",
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(facecolor=col, alpha=0.6, edgecolor='none', pad=1), ha='left')

    if start_times:
        ax.plot([monitor_ch] * len(start_times), start_times, 'r.', ms=6, alpha=0.6,
                label=f'Detections ({len(start_times)})')

    if valid:
        speeds = [r['speed_kmh'] for r in valid]
        ins = ax.inset_axes([0.75, 0.03, 0.23, 0.25])
        ins.hist(speeds, bins=max(3, min(20, len(speeds))), color='#42a5f5', edgecolor='white', lw=1.0)
        ins.set_xlabel('Speed (km/h)', fontsize=10, color='white', fontweight='bold')
        ins.set_ylabel('Count', fontsize=10, color='white', fontweight='bold')
        ins.tick_params(colors='white', labelsize=9)
        ins.set_facecolor('#1a1a2e')
        ins.grid(axis='y', alpha=0.3, ls='--')
        for sp in ins.spines.values():
            sp.set_edgecolor('#444')

    ax.set_title(f"10-minute chunk — {len(valid)} vehicles (solid = RTS track, dashed = predicted)"
                 f"\n{title_info}", fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_xlabel("Channel (relative)", fontsize=14, color='white')
    ax.set_ylabel(f"Time (samples / {fs:.0f} Hz)", fontsize=14, color='white')
    ax.tick_params(colors='white')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='white')
    ax.set_facecolor('#0f172a')
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
