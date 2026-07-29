# Data directory

This repository stores datasets under `data/`.

## EUR/USD inputs

- `data/EURUSD/raw_norm_EURUSD.csv` — normalized price series used by the EUR/USD experiments.
- `data/EURUSD/EUR-USD_Minute_*.csv` — local minute-level data files used for additional experiments.

## Notes

- `raw_norm_EURUSD.csv` is tracked with Git LFS.
- To install Git LFS and fetch tracked files:
  ```bash
git lfs install
git lfs pull
  ```
- Keep local or sensitive data files under `data/EURUSD/` but do not commit them unless they are intended for the repository.
