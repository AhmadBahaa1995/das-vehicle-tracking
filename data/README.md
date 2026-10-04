# Sample data

The sample is too large for GitHub. It is archived on Zenodo and must be downloaded manually:

**https://doi.org/10.5281/zenodo.23130284**

It is the ten-minute development window of the manuscript: 2 February 2024, 09:00–09:10 JST (00:00–00:10 UTC). It covers the 500 channels of the ~1-km study segment, numbered locally from 0 to 500, and is stored as a single HDF5 file.

## Setup

1. Download the `.h5` file from the Zenodo record above.
2. Put it in this folder:

```
data/
├── README.md
└── <sample>.h5      the downloaded file (any name)
```

3. From the repository root, run:

```bash
python scripts/run_window.py --input data \
    --first-channel 0 --last-channel 500 --monitor-ch 190 --out results/sample --plots
```

`--input data` reads every `.h5` file directly in this folder, so the file name does not matter. Keep only the sample file here.

## File contents

The file contains only the raw data array (time × channel, sampled at 1000 Hz) and the timestamp of each sample. It has no other metadata. The readers handle this as follows:

| Quantity | Source |
|---|---|
| Sampling rate | inferred from the timestamps |
| Start time | first timestamp (UTC) |
| Channel spacing | `DX = 2.0419 m` in `dastrack/params.py` |
| Channel range | the whole file: `--first-channel 0 --last-channel 500` |

Everything in this folder except this README is ignored by git.
