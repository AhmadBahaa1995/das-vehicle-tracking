"""
Kalman-fused F-K slant-stack vehicle tracker (manuscript "Kalman-filtered
tracking with RTS smoothing").

For each detection at the monitor channel:

1. **Bootstrap** - an F-K slant-stack over a wide patch proposes candidate
   velocities; the top ``n_seeds`` are kept, with both directions guaranteed.
2. **Track** - each seed is tracked forward and backward in time by a Kalman
   filter.  A narrow window slides along the KF prediction; each window gives
   an F-K velocity measurement (gated) and a de-slant centroid position
   measurement (gated, with position skepticism).  A window is *alive* only if
   the F-K contrast, the centroid distance and the local energy all pass; the
   track ends after ``max_consecutive_misses`` dead windows.
3. **Score** - ``score = sqrt(P_norm) * mean_energy_alive * n_updates * alive_frac``
   (Eq. 16); the highest-scoring seed is the vehicle's track.
4. **Smooth** - an RTS pass over the merged track gives the speed profile and
   its uncertainty.  Tracks with too few updates, near-zero travel distance or
   a speed outside the valid range are rejected.

Matrices are (time x channel); channels are relative to the loaded slab.
"""
import numpy as np

from .detection import detect_entries
from .fk import fk_candidates, slant_stack
from .kalman import KF


# ─────────────────────────────────────────────────────────────────────────────
# Window helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rms(patch, centre_half=3):
    nc = patch.shape[1]
    lo = max(0, nc // 2 - centre_half)
    hi = min(nc, nc // 2 + centre_half + 1)
    return float(np.sqrt(np.mean(patch[:, lo:hi] ** 2)))


def _peak_energy_channel(matrix, t_center, monitor_ch, fs,
                         half_ch=5, half_t_sec=5.0, max_ch_shift=5):
    """
    Refine the spatial anchor near the detection point, clamped to within
    ``max_ch_shift`` channels of the monitor channel so it cannot jump onto a
    neighbouring vehicle's streak.
    """
    m, n = matrix.shape
    t_lo = max(0, t_center - int(half_t_sec * fs))
    t_hi = min(m, t_center + int(half_t_sec * fs) + 1)
    ch_lo = max(0, monitor_ch - half_ch)
    ch_hi = min(n, monitor_ch + half_ch + 1)
    col_e = np.sum(matrix[t_lo:t_hi, ch_lo:ch_hi] ** 2, axis=0)
    if col_e.max() <= 0:
        return float(monitor_ch)
    top = np.argsort(col_e)[-3:]
    ch = ch_lo + float(np.average(top, weights=col_e[top]))
    return float(np.clip(ch, monitor_ch - max_ch_shift, monitor_ch + max_ch_shift))


def extract_patch(matrix, t_center, ch_center, half_time_sec, half_space_ch, fs):
    m, n = matrix.shape
    half_t = int(half_time_sec * fs)
    t_lo = max(0, t_center - half_t)
    t_hi = min(m - 1, t_center + half_t)
    ch_lo = max(0, ch_center - half_space_ch)
    ch_hi = min(n - 1, ch_center + half_space_ch)
    return matrix[t_lo:t_hi + 1, ch_lo:ch_hi + 1].copy(), t_lo, t_hi, ch_lo, ch_hi


def deslant_centroid(patch, ch_lo, v_ms, fs, dx, top_frac=0.8):
    """
    De-slant position measurement (Eqs. 17-18): shear each time row by the
    current velocity so the diagonal streak collapses to a vertical ridge, then
    return the energy-weighted column centroid as an absolute channel index.
    """
    nt, nx = patch.shape
    if nt < 4 or nx < 4:
        return None
    rows = np.arange(nt) - nt // 2
    shifts = (v_ms * (rows / fs) / dx)
    de = np.empty_like(patch)
    for r in range(nt):
        de[r] = np.roll(patch[r], -int(round(shifts[r])))
    col_e = np.sum(de ** 2, axis=0)
    if col_e.max() <= 0:
        return None
    msk = col_e >= np.quantile(col_e, top_frac)
    if not np.any(msk):
        return None
    return ch_lo + float(np.average(np.arange(nx)[msk], weights=col_e[msk]))


# ─────────────────────────────────────────────────────────────────────────────
# One-direction tracking core
# ─────────────────────────────────────────────────────────────────────────────

def _track_direction(matrix, fs, dx, t_start, ch_start, v_seed_ms, n_ch, direction,
                     step_samples, half_t_sec, half_ch, v_min_mps, v_max_mps,
                     gate_sigma, gate_lo_ms, gate_hi_ms, env_rms_threshold,
                     min_contrast, max_consecutive_misses, max_steps,
                     max_ch_jump=10.0, min_local_energy=0.015,
                     q_vel=0.8, r_vel=6.25, contrast_r_lo=0.3, contrast_r_hi=8.0,
                     pos_skeptic=3.0):
    kf = KF(ch_start, v_seed_ms, fs, dx, v_min=v_min_mps, v_max=v_max_mps,
            q_vel=q_vel, r_vel=r_vel, contrast_r_lo=contrast_r_lo, contrast_r_hi=contrast_r_hi)
    rec = dict(t=[], ch=[], v=[], raw=[], rank=[], alive=[], local_e=[], n_upd=0)
    miss = 0
    t_cur = t_start
    for _ in range(max_steps):
        t_cur += direction * step_samples
        if t_cur < half_t_sec * fs or t_cur > matrix.shape[0] - half_t_sec * fs:
            break
        pred_ch = kf.predict(direction * step_samples)
        pci = int(np.clip(pred_ch, half_ch + 1, n_ch - half_ch - 2))
        patch, _, _, ch_lo, _ = extract_patch(matrix, t_cur, pci, half_t_sec, half_ch, fs)

        alive = False
        sel_v = None
        sel_rank = -1
        # local energy at the predicted channel (+/-3 ch): is the vehicle still here?
        ti = int(t_cur)
        local_e = float(np.mean(matrix[ti, max(0, pci - 3):pci + 4] ** 2)) \
            if 0 <= ti < matrix.shape[0] else 0.0

        if patch.size >= 16 and _rms(patch) >= env_rms_threshold:
            v_ax, sc, _, _, _ = slant_stack(patch, fs, dx, v_max_mps=v_max_mps + 5)
            cands, contrast = fk_candidates(v_ax, sc)
            if contrast >= min_contrast and local_e >= min_local_energy:
                gate = kf.gate_ms(gate_sigma, gate_lo_ms, gate_hi_ms)
                for rank, (v, p) in enumerate(cands):
                    if abs(v - kf.v_ms) <= gate and v_min_mps <= abs(v) <= v_max_mps:
                        sel_v, sel_rank = v, rank
                        break
                if sel_v is not None:
                    r_scale = float(np.clip(3.0 / max(contrast, 1e-3), kf.cr_lo, kf.cr_hi))
                    kf.update_velocity(sel_v, r_scale=r_scale)
                    # De-slant position correction, accepted only if the ridge is
                    # within the trust radius of the prediction.
                    ch_meas = deslant_centroid(patch, ch_lo, kf.v_ms, fs, dx)
                    if ch_meas is not None and abs(ch_meas - pred_ch) <= max_ch_jump:
                        # Trust the pull less for faint windows and for centroids that
                        # disagree with the prediction (position skepticism).
                        disagree = abs(ch_meas - pred_ch) / max(max_ch_jump, 1e-3)
                        pos_r = r_scale * (1.0 + pos_skeptic * disagree)
                        kf.update_position(ch_meas, r_scale=pos_r)
                        rec['n_upd'] += 1
                        alive = True
                        miss = 0
                    else:
                        miss += 1      # ridge too far: energy belongs to another object
                else:
                    miss += 1
            else:
                miss += 1
        else:
            miss += 1

        kf.commit()
        rec['t'].append(t_cur / fs)
        rec['ch'].append(kf.ch)
        rec['v'].append(kf.v_ms * 3.6)
        rec['raw'].append(sel_v * 3.6 if sel_v is not None else None)
        rec['rank'].append(sel_rank)
        rec['alive'].append(alive)
        rec['local_e'].append(local_e if alive else 0.0)
        if miss >= max_consecutive_misses:
            break

    rec['kf'] = kf
    return rec


def _trim_to_alive(rec):
    if not any(rec['alive']):
        return rec
    last = max(i for i, a in enumerate(rec['alive']) if a)
    for k in ['t', 'ch', 'v', 'raw', 'rank', 'alive', 'local_e']:
        rec[k] = rec[k][:last + 1]
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# One vehicle
# ─────────────────────────────────────────────────────────────────────────────

def track_vehicle(t_detection, matrix, fs, dx, monitor_channel,
                  kf_bootstrap_half_time=5.0, kf_bootstrap_half_ch=60,
                  kf_window_sec=2.0, kf_half_space_ch=20, kf_slide_overlap=0.5,
                  v_min_kmh=10, v_max_kmh=80,
                  gate_sigma=3.0, gate_lo_kmh=6.0, gate_hi_kmh=22.0,
                  env_rms_threshold=0.03, min_contrast=2.2,
                  max_consecutive_misses=2,
                  max_ch_jump=10.0, min_local_energy=0.015,
                  anchor_half_ch=5, anchor_half_t_sec=5.0, anchor_max_shift=5,
                  q_vel=0.8, r_vel=6.25, contrast_r_lo=0.3, contrast_r_hi=8.0,
                  min_track_updates=3, min_track_dist_m=10.0, pos_skeptic=3.0,
                  track_back_max_sec=30.0, track_forward_max_sec=30.0,
                  n_seeds=3, car_id=None, verbose=False, **_ignored):
    """
    Track the vehicle detected at sample ``t_detection``.

    Returns a result dict (see keys below), or ``None`` if the bootstrap patch
    is too small or has no F-K peak.  ``result['valid']`` is False for tracks
    rejected by the self-rejection rules.

    Main keys: ``t_sec`` (detection time), ``velocity_kmh`` (signed, at the
    monitor channel), ``speed_kmh``, ``direction`` (+1 toward higher channel,
    -1 toward lower), ``valid``, ``n_updates``, ``t_entry_sec``/``t_exit_sec``,
    ``ch_entry``/``ch_exit``, ``distance_m``, ``duration_sec``,
    ``speed_min_kmh``/``speed_max_kmh``, ``kf_uncertainty_kmh``,
    ``profile_t_sec``/``track_ch``/``speed_profile_kmh``/``speed_sigma_kmh``
    (RTS-smoothed track), ``bootstrap_kmh`` (strongest single-patch F-K peak)
    and ``diag`` (arrays for :func:`dastrack.plotting.plot_vehicle_diagnostic`).
    """
    log = print if verbose else (lambda *a, **k: None)
    m, n = matrix.shape
    v_min = v_min_kmh / 3.6
    v_max = v_max_kmh / 3.6
    gate_lo = gate_lo_kmh / 3.6
    gate_hi = gate_hi_kmh / 3.6
    step = max(1, int(int(kf_window_sec * fs) * 2 * (1.0 - kf_slide_overlap)))
    t_det_s = t_detection / fs

    log(f"\n{'─' * 60}\n  CAR {car_id}  |  detected @ {t_det_s:.2f} s\n{'─' * 60}")

    # 1. bootstrap F-K over a wide patch at the monitor channel
    patch0, *_ = extract_patch(matrix, t_detection, monitor_channel,
                               kf_bootstrap_half_time, kf_bootstrap_half_ch, fs)
    if patch0.size < 64:
        log("  x  bootstrap patch too small.")
        return None
    v_ax0, sc0, fk_db0, freqs0, wvn0 = slant_stack(patch0, fs, dx, v_max_mps=v_max + 5)
    all_cands, _ = fk_candidates(v_ax0, sc0, top_k=6)
    if not all_cands:
        log("  x  no bootstrap peak.")
        return None
    ch_at_det = _peak_energy_channel(matrix, t_detection, monitor_channel, fs,
                                     half_ch=anchor_half_ch, half_t_sec=anchor_half_t_sec,
                                     max_ch_shift=anchor_max_shift)
    log("  Bootstrap peaks: " + ", ".join(f"{v * 3.6:+.0f}({p:.1e})" for v, p in all_cands[:5]))

    # seeds: top-n_seeds by power, plus guarantee both directions are present
    seeds = list(all_cands[:n_seeds])
    if all(v < 0 for v, _ in seeds):
        pos = [c for c in all_cands if c[0] > 0]
        if pos:
            seeds.append(pos[0])
    if all(v > 0 for v, _ in seeds):
        neg = [c for c in all_cands if c[0] < 0]
        if neg:
            seeds.append(neg[0])
    log(f"  Seeds tried (km/h): {[round(v * 3.6) for v, _ in seeds]}")

    # 2-3. track each seed both ways and score it (coherence-weighted)
    common = (kf_window_sec, kf_half_space_ch, v_min, v_max, gate_sigma, gate_lo, gate_hi,
              env_rms_threshold, min_contrast, max_consecutive_misses)
    tail = (max_ch_jump, min_local_energy, q_vel, r_vel, contrast_r_lo, contrast_r_hi,
            pos_skeptic)
    pmax = max(p for _, p in all_cands)
    best = None
    seed_scores = []
    for v_seed, p_seed in seeds:
        fwd = _trim_to_alive(_track_direction(
            matrix, fs, dx, t_detection, ch_at_det, v_seed, n, +1, step, *common,
            int(track_forward_max_sec / (step / fs)), *tail))
        bwd = _trim_to_alive(_track_direction(
            matrix, fs, dx, t_detection, ch_at_det, v_seed, n, -1, step, *common,
            int(track_back_max_sec / (step / fs)), *tail))
        n_upd = fwd['n_upd'] + bwd['n_upd']
        all_alive = fwd['alive'] + bwd['alive']
        alive_frac = (sum(all_alive) / len(all_alive)) if all_alive else 0.0
        e_alive = [e for e, a in zip(fwd['local_e'] + bwd['local_e'], all_alive) if a]
        mean_e = float(np.mean(e_alive)) if e_alive else 0.0
        pnorm = p_seed / pmax
        score = (pnorm ** 0.5) * mean_e * n_upd * alive_frac
        seed_scores.append(dict(v_kmh=v_seed * 3.6, pnorm=pnorm, n_updates=n_upd,
                                alive_frac=alive_frac, mean_energy=mean_e, score=score))
        log(f"    seed {v_seed * 3.6:+6.0f} km/h -> Pnorm={pnorm:.2f} upd={n_upd:2d} "
            f"aliveFrac={alive_frac:.2f} E={mean_e:.4f}  score={score:.4f}")
        if best is None or score > best['score']:
            best = dict(score=score, v_seed=v_seed, fwd=fwd, bwd=bwd)

    fwd, bwd, v_seed = best['fwd'], best['bwd'], best['v_seed']
    log(f"  -> WINNER seed {v_seed * 3.6:+.0f} km/h (score {best['score']:.4f})")

    # merge backward + forward chronologically
    trk = dict(t=list(reversed(bwd['t'])) + fwd['t'],
               ch=list(reversed(bwd['ch'])) + fwd['ch'],
               alive=list(reversed(bwd['alive'])) + fwd['alive'],
               raw=list(reversed(bwd['raw'])) + fwd['raw'],
               rank=list(reversed(bwd['rank'])) + fwd['rank'])

    base = dict(car_id=car_id, t_sample=int(t_detection), t_sec=t_det_s, ch_det=ch_at_det,
                bootstrap_kmh=all_cands[0][0] * 3.6, seed_kmh=v_seed * 3.6,
                seed_scores=seed_scores)

    if not any(trk['alive']):
        log("  !  no live windows.")
        final_kmh = v_seed * 3.6
        return dict(base, velocity_kmh=final_kmh, speed_kmh=abs(final_kmh),
                    direction=1 if final_kmh >= 0 else -1, valid=False, n_updates=0,
                    t_entry_sec=t_det_s, t_exit_sec=t_det_s,
                    ch_entry=ch_at_det, ch_exit=ch_at_det, distance_m=0.0, duration_sec=0.0,
                    speed_min_kmh=abs(final_kmh), speed_max_kmh=abs(final_kmh),
                    kf_uncertainty_kmh=0.0, profile_t_sec=[t_det_s], track_ch=[ch_at_det],
                    speed_profile_kmh=[final_kmh], speed_sigma_kmh=[0.0], diag=None)

    alive_idx = [i for i, a in enumerate(trk['alive']) if a]
    t_entry = trk['t'][alive_idx[0]]
    t_exit = trk['t'][alive_idx[-1]]
    ch_entry = trk['ch'][alive_idx[0]]
    ch_exit = trk['ch'][alive_idx[-1]]

    # 4. RTS smoothing over the ordered timeline, with the detection anchor
    # pinned so the smoothed track passes through the detection point.
    order = np.argsort(trk['t'])
    t_list = [trk['t'][i] for i in order]
    raw_list = [trk['raw'][i] for i in order]
    ch_list = [trk['ch'][i] for i in order]
    rank_list = [trk['rank'][i] for i in order]
    if t_det_s not in t_list:
        ins = int(np.searchsorted(t_list, t_det_s))
        t_list.insert(ins, t_det_s)
        raw_list.insert(ins, v_seed * 3.6)      # velocity prior at detection
        ch_list.insert(ins, ch_at_det)          # pinned channel at detection
        rank_list.insert(ins, -1)
    t_ord = np.array(t_list)
    kf2 = KF(ch_list[0], v_seed, fs, dx, v_min=v_min, v_max=v_max,
             q_vel=q_vel, r_vel=r_vel, contrast_r_lo=contrast_r_lo, contrast_r_hi=contrast_r_hi)
    for j in range(len(t_ord)):
        ds = step if j == 0 else max(1, int(round((t_ord[j] - t_ord[j - 1]) * fs)))
        kf2.predict(ds)
        if raw_list[j] is not None:
            kf2.update_velocity(raw_list[j] / 3.6)
            kf2.update_position(ch_list[j])
        kf2.commit()
    xs, Ps = kf2.rts()
    sm_v = xs[:, 1] * 3.6
    sm_ch = xs[:, 0]
    sm_sig = np.sqrt(np.clip(Ps[:, 1, 1], 0, None)) * 3.6

    idx_det = int(np.argmin(np.abs(t_ord - t_det_s)))
    final_kmh = float(sm_v[idx_det])
    n_upd = sum(1 for r in raw_list if r is not None)
    dist_m = abs(ch_exit - ch_entry) * dx
    # self-rejection: a real pass moves along the fibre and survives a few windows
    valid = bool(v_min_kmh <= abs(final_kmh) <= v_max_kmh
                 and n_upd >= min_track_updates
                 and dist_m >= min_track_dist_m)
    direction = int(np.sign(final_kmh)) if final_kmh != 0 else 1

    log(f"  Lifecycle: entry t={t_entry:.1f}s ch={ch_entry:.0f} -> "
        f"exit t={t_exit:.1f}s ch={ch_exit:.0f}  (on road {t_exit - t_entry:.1f}s, {dist_m:.0f} m)")
    log(f"  Speed @ monitor: {final_kmh:+.1f} km/h | profile "
        f"{np.abs(sm_v).min():.1f}-{np.abs(sm_v).max():.1f} km/h | updates {n_upd}")
    log(f"  Direction {'-> (+)' if direction > 0 else '<- (-)'} | valid: {'yes' if valid else 'no'}"
        + ("" if valid else "  (rejected: too short / 0 m / out of band)"))

    return dict(base, velocity_kmh=final_kmh, speed_kmh=abs(final_kmh),
                direction=direction, valid=valid, n_updates=n_upd,
                t_entry_sec=t_entry, t_exit_sec=t_exit, ch_entry=ch_entry, ch_exit=ch_exit,
                distance_m=dist_m, duration_sec=t_exit - t_entry,
                speed_min_kmh=float(np.abs(sm_v).min()),
                speed_max_kmh=float(np.abs(sm_v).max()),
                kf_uncertainty_kmh=float(sm_sig[idx_det]),
                profile_t_sec=t_ord.tolist(), track_ch=sm_ch.tolist(),
                speed_profile_kmh=sm_v.tolist(), speed_sigma_kmh=sm_sig.tolist(),
                diag=dict(all_cands=all_cands, patch0=patch0, fk_db0=fk_db0,
                          freqs0=freqs0, wvn0=wvn0, v_ax0=v_ax0, sc0=sc0,
                          raw_kmh=raw_list, rank=rank_list, idx_det=idx_det,
                          monitor_channel=monitor_channel,
                          v_min_kmh=v_min_kmh, v_max_kmh=v_max_kmh))


# ─────────────────────────────────────────────────────────────────────────────
# Whole window
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(matrix, fs, dx, monitor_ch, min_height=0.30, min_dist_sec=4.0,
                 v_min_kmh=10, v_max_kmh=80, tracker_params=None,
                 verbose=True, show_plots=False, plot_dir=None):
    """
    Detect and track every vehicle in a denoised (time x channel) matrix.

    Parameters
    ----------
    tracker_params : dict
        Tracker settings, e.g. :data:`dastrack.params.TRACKER_DEVELOPMENT`.
    show_plots, plot_dir
        Draw the entry-detection figure and one diagnostic figure per vehicle;
        figures are saved to ``plot_dir`` when it is given.

    Returns ``(starts, results, valid_results)``.
    """
    tracker_params = dict(tracker_params or {})
    starts, smoothed = detect_entries(matrix, fs=fs, monitor_channel=monitor_ch,
                                      monitor_width=5, cutoff_hz=1.5,
                                      min_height=min_height, min_distance_sec=min_dist_sec,
                                      verbose=verbose)
    if show_plots or plot_dir:
        from . import plotting
        plotting.plot_entry_detection(matrix, smoothed, starts, monitor_ch, min_height, fs,
                                      save_path=plotting.out_path(plot_dir, 'entry_detection.png'),
                                      show=show_plots)

    results = []
    for i, t0 in enumerate(starts):
        r = track_vehicle(t0, matrix, fs, dx, monitor_ch, car_id=i + 1,
                          v_min_kmh=v_min_kmh, v_max_kmh=v_max_kmh,
                          verbose=verbose, **tracker_params)
        results.append(r)
        if r is not None and r['diag'] is not None and (show_plots or plot_dir):
            plotting.plot_vehicle_diagnostic(
                r, matrix, fs, dx,
                save_path=plotting.out_path(plot_dir, f'vehicle_{i + 1:03d}.png'),
                show=show_plots)

    valid = [r for r in results if r and r['valid']]
    if verbose:
        print_report(results)
    return starts, results, valid


def print_report(results):
    """Text summary of a window's results."""
    valid = [r for r in results if r and r['valid']]
    invalid = [r for r in results if r and not r['valid']]
    print(f"\n{'=' * 84}\n  FINAL REPORT\n{'=' * 84}")
    print(f"  Detections {len(results)}  Valid {len(valid)}  Rejected {len(invalid)}")
    print(f"\n  {'ID':>3} {'t(s)':>6} {'Speed':>6} {'Range':>10} {'Dir':>4} "
          f"{'tEntry':>6} {'tExit':>6} {'chIn':>5} {'chOut':>5} {'Dist_m':>6} {'Upd':>4}")
    print(f"  {'-' * 80}")
    for r in valid:
        rng = f"{r['speed_min_kmh']:.0f}-{r['speed_max_kmh']:.0f}"
        print(f"  {r['car_id']:>3} {r['t_sec']:>6.1f} {r['speed_kmh']:>6.1f} {rng:>10} "
              f"{'+' if r['direction'] > 0 else '-':>4} {r['t_entry_sec']:>6.1f} {r['t_exit_sec']:>6.1f} "
              f"{r['ch_entry']:>5.0f} {r['ch_exit']:>5.0f} {r['distance_m']:>6.0f} {r['n_updates']:>4}")
    if invalid:
        print("\n  Rejected:")
        for r in invalid:
            print(f"    Car {r['car_id']} @ {r['t_sec']:.1f}s -> {r['speed_kmh']:.1f} km/h")
    if valid:
        s = [r['speed_kmh'] for r in valid]
        print(f"\n  Speed at monitor: min {min(s):.1f} max {max(s):.1f} mean {np.mean(s):.1f} "
              f"median {np.median(s):.1f} km/h")
    print(f"{'=' * 84}")


RESULT_COLUMNS = ['car_id', 't_sec', 'velocity_kmh', 'speed_kmh', 'direction',
                  'distance_m', 'duration_sec', 'speed_min_kmh', 'speed_max_kmh',
                  'n_updates', 't_entry_sec', 't_exit_sec', 'ch_entry', 'ch_exit',
                  'kf_uncertainty_kmh', 'bootstrap_kmh', 'valid']


def results_table(results):
    """Results as a pandas DataFrame (one row per detection, scalar columns only)."""
    import pandas as pd
    rows = [{k: r[k] for k in RESULT_COLUMNS} for r in results if r]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
