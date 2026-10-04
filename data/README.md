# Sample data

The sample dataset is too large for GitHub. It is archived on Zenodo and must be downloaded manually:

**https://doi.org/10.5281/zenodo.XXXXXXX**

The sample is the ten-minute development window of the manuscript: 2 February 2024, 09:00–09:10 JST (00:00–00:10 UTC), covering the ~1-km study segment.

## Setup

1. Download the archive from the Zenodo record above.
2. Extract it into this folder. The result should look like this:

```
data/
└── sample/
    └── 2024/02/02/00/      hour folder (UTC)
        ├── 00/*.h5         one folder per minute
        ├── 01/*.h5
        ├── ...
        └── 09/*.h5
```

3. Run the pipeline from the repository root:

```bash
python scripts/run_window.py --data-root data/sample --date 2024/02/02 --hour 0 --minutes 0 10 \
    --first-channel 0 --last-channel 500 --monitor-ch 190 --out results/sample --plots
```

The files keep the original HDF5 structure (`Acquisition/Raw[0]/RawData`, `RawDataTime`). Only the 500 channels of the study segment are included, numbered locally from 0 to 500.

Everything in this folder except this README is ignored by git.
