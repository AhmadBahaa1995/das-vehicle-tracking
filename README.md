# dastrack — Kalman-fused F-K slant-stack vehicle tracking on telecom-fiber DAS

Code accompanying

> Ahmad, A. B., T. Tsuji, and M. Karrenbach. *Urban Traffic Kinematics from Telecommunication Fiber: Kalman-Fused F-K Slant-Stacking of DAS Data.*

It is a training-free pipeline that recovers each vehicle's **speed, direction and full trajectory with uncertainty** from distributed acoustic sensing (DAS) data recorded on buried telecommunication fiber. It also contains the **parallel production driver** used to catalog one month of continuous data.

```
raw DAS ──► preprocessing ──► detection at ──► F-K slant-stack seeds ──► Kalman tracking ──► RTS smoothing ──► vehicle catalog
            (70–100 Hz,        a monitor        (multi-seed, both        (F-K velocity +       + self-rejection
             |.|, median,      channel           directions)              de-slant position,
             norm, wavelet)                                                gated lifecycle)
```

## Method in brief

| Stage | What it does | Code |
|---|---|---|
| Preprocessing (Eqs. 1–4) | 4th-order zero-phase Butterworth 70–100 Hz, 1 s edge trim → \|strain\| → mute bad channels → per-channel median subtraction → 2–98 percentile normalisation → db4 wavelet denoise (2 levels, hybrid hard/soft threshold, λ = 0.15, a = 0.5) | `dastrack/preprocessing.py` |
| Detection | Mean of 5 channels at the monitor channel → 1.5 Hz low-pass envelope → peak search (threshold 0.30, ≥ 4 s apart) | `dastrack/detection.py` |
| F-K measurement (Eqs. 5–9) | 2-D Hann taper → F-K power → slant-stack scan of mean power along *f = −vk* for 360 candidate velocities, \|f\| ≤ 0.08 Hz excluded | `dastrack/fk.py` |
| Kalman tracking (Eqs. 10–15) | State [channel, velocity]; constant-velocity model; F-K velocity updates (gated, contrast-scaled noise) and de-slant centroid position updates (trust radius 13 ch, position skepticism 3.0) | `dastrack/kalman.py`, `dastrack/tracker.py` |
| Multi-seed bootstrap (Eq. 16) | Top-3 bootstrap peaks + guaranteed opposite direction; score = √P_norm · mean on-track energy · n_updates · alive fraction | `dastrack/tracker.py` |
| Lifecycle + RTS (Eqs. 17–19) | Window alive only if F-K contrast ≥ 1.5, centroid within the trust radius and local energy above the floor; 2 dead windows end the track; RTS backward pass; reject tracks with < 3 updates, < 10 m travel, or speed outside 10–80 km/h | `dastrack/tracker.py` |
| Production run | Date range → hours → worker processes → 10-minute chunks → catalog CSV + per-chunk QC images | `dastrack/production.py` |

**Sign convention.** Velocity > 0 means the vehicle moves toward higher channel index (`direction = +1`). Such a streak lies on the line *f = −vk* in the F-K plane.

**Calibration.** Every reported speed scales linearly with the channel spacing `DX` (2.0419 m, from the interrogator metadata).

## Repository layout

```
dastrack/            the pipeline as an importable package
  io.py              HDF5 readers (fast window loader; retry-guarded production reader)
  preprocessing.py   shared preprocessing chain
  detection.py       monitor-channel detection
  fk.py              F-K slant-stack, candidate peaks
  kalman.py          Kalman filter + RTS smoother
  tracker.py         per-vehicle tracking, window pipeline, results table
  params.py          parameter sets (development window / month-scale production)
  production.py      parallel month-scale driver, catalog writer
  catalog.py         month-catalog statistics and figure
  plotting.py        preprocessing, detection, per-vehicle diagnostic, QC figures
scripts/
  run_window.py              one continuous window (e.g. the 10-min development window)
  run_month.py               month-scale parallel production run → catalog CSV
  plot_month_catalog.py      catalog statistics + Figure 8
  plot_weekly_diagnostics.py random per-vehicle diagnostics, one per week (Figure 9)
notebooks/
  walkthrough.ipynb          step-by-step walkthrough of the method on one window
tests/                       end-to-end tests on the sample data
data/                        put the downloaded sample file here (see data/README.md)
```

## Installation

Python ≥ 3.9.

```bash
git clone https://github.com/AhmadBahaa1995/das-vehicle-tracking.git
cd das-vehicle-tracking
pip install -r requirements.txt     # or: pip install -e .
```

## Data format

The full archive is stored as fixed-length HDF5 segments in per-minute folders:

```
<data_root>/<YYYY>/<MM>/<DD>/<HH>/<MM>/*.h5        (folder times in UTC)
```

Each file holds an `Acquisition/Raw[0]` group (or `Acquisition`) with

* `RawData` — strain-rate/phase samples, time × channel or channel × time (detected automatically),
* `RawDataTime` — sample times in microseconds since the Unix epoch.

The native sampling rate is read from the `PulseRate` or `AcquisitionFrequency` attribute. Without those attributes it is inferred from the timestamps, falling back to 1000 Hz.

Files without this group structure or metadata, such as the sample, are also accepted. The raw data is then taken to be the largest 2-D array in the file, and the timestamps the 1-D array matching its time axis, in s, ms, µs or ns since the epoch. Such files are loaded with `--input` (a file, or a folder of files that form one continuous window) instead of `--data-root/--date/--hour`.

Only the channel range `--first-channel:--last-channel` is read. All channel numbers inside the pipeline, including `--monitor-ch`, are **relative to that slab**.

## Quick start with the sample data

A ten-minute sample, the development window of the manuscript, is archived on Zenodo. It is too large for GitHub, so download it manually:

**Sample data: https://doi.org/10.5281/zenodo.23130284**

Put the downloaded `.h5` file in the `data/` folder (details in [data/README.md](data/README.md)) and run:

```bash
python scripts/run_window.py --input data \
    --first-channel 0 --last-channel 500 --monitor-ch 190 --out results/sample --plots
```

This writes `results/sample/vehicles.csv` (one row per detection; `valid` marks the reported vehicles) and, in `results/sample/figures/`, the preprocessing and detection figures plus one diagnostic figure per vehicle.

The sample is a single file holding only the raw data of the 500 study-segment channels and a timestamp per sample. The sampling rate is inferred from the timestamps, and the channel spacing comes from `DX` in `dastrack/params.py`.

The walkthrough notebook (`notebooks/walkthrough.ipynb`) runs the method step by step on the same file.

## Reproducing the paper

The study segment is ~1 km long: 500 channels (local channels 0–500), with the monitor channel at local channel 190. The full archive is not public (see *Data availability*). With access to it, replace `<CH0>` and `<CH1> = <CH0> + 500` with the absolute channel range of the segment. With the sample, the development-window run is the quick-start command above.

**Development window** (Method section, Figures 3–5 and 7: 2 February 2024, 09:00–09:10 JST = 00:00–00:10 UTC):

```bash
python scripts/run_window.py --data-root /path/to/archive --date 2024/02/02 --hour 0 --minutes 0 10 \
    --first-channel <CH0> --last-channel <CH1> --monitor-ch 190 --params development \
    --out results/development --plots
```

**Month-scale production run** (Figures 8–9, Table 4: 1–28 February 2024, 672 hours):

```bash
python scripts/run_month.py --data-root /path/to/archive --start 2024-02-01 --end 2024-02-28 \
    --first-channel <CH0> --last-channel <CH1> --monitor-ch 190 --workers 8 --out results/month

python scripts/plot_month_catalog.py --csv results/month/catalog_2024-02-01_to_2024-02-28.csv \
    --start 2024-02-01 --end 2024-02-28 --out results/month/figure8_catalog.png

python scripts/plot_weekly_diagnostics.py --data-root /path/to/archive --start 2024-02-01 --end 2024-02-28 \
    --first-channel <CH0> --last-channel <CH1> --monitor-ch 190 --out results/month/weekly_diagnostics
```

The production driver splits each hour into continuous 10-minute chunks of 20 × 30-s segments, which keeps memory use bounded. It skips unreadable or incomplete files and writes one QC image per chunk. Eight workers were used, a number set by the throughput of the 1 GbE link to the storage array rather than by CPU count.

### Catalog columns

| column | meaning |
|---|---|
| `time` | crossing time at the monitor channel, local time (UTC + `--utc-offset`, default 9 h = JST) |
| `speed` | signed speed at the monitor channel from the RTS-smoothed track (km/h; + toward higher channel) |
| `direction` | +1 / −1 |
| `distance_m`, `duration_sec` | on-road travel distance and duration between track entry and exit |
| `speed_min_kmh`, `speed_max_kmh` | range of the smoothed speed profile |
| `n_updates` | accepted F-K measurement windows |
| `ch_entry`, `ch_exit` | entry and exit channel (relative to the slab) |
| `uncertainty_kmh` | RTS 1-σ speed uncertainty at the monitor channel |

## Parameters

Detection and validity settings (`dastrack.params.DETECTION`) are shared by every run. The tracker has two parameter sets:

| Parameter | `TRACKER_DEVELOPMENT` | `TRACKER_PRODUCTION` |
|---|---|---|
| Velocity gate floor / ceiling | 25 / 35 km/h | 20 / 45 km/h |
| Process noise `q_vel` | 0.8 | 0.4 |
| Velocity measurement noise `r_vel` | 6.25 | 9.0 |
| *All other settings* | *identical, see `dastrack/params.py`* | |

`TRACKER_DEVELOPMENT` was used on the development window. `TRACKER_PRODUCTION` was used for the February 2024 month run; its smaller process noise gives smoother tracks for unattended operation.

## Tests

```bash
pip install pytest
pytest tests
```

The tests run the window pipeline and the production worker on the sample file. Among other things they check the sign convention: positive velocity means the track moves toward higher channel index. They are skipped if no sample file is in `data/`.

## Data availability

The DAS data were recorded on a commercial telecommunication cable in Tokyo, Japan. They are proprietary and were made available to the authors under a confidentiality agreement with the cable operator. The full archive cannot be released. A ten-minute sample of the study segment (the development window) is available on Zenodo for testing the code: https://doi.org/10.5281/zenodo.23130284.

## Citation

If you use this code, please cite the article above (see also `CITATION.cff`).

## License

MIT — see `LICENSE`.
