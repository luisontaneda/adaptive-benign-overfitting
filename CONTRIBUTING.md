# Contributing to Adaptive Benign Overfitting

Thank you for contributing to this research codebase. These guidelines help keep the repository easy to use and maintain.

## Getting started

1. Build the project:
   ```bash
   make -j
   ```
2. Build the test binaries:
   ```bash
   make tests
   ```
3. Run all unit tests:
   ```bash
   make check
   ```
4. List available build targets:
   ```bash
   make help
   ```

## Data

- `data/EURUSD/raw_norm_EURUSD.csv` is tracked with Git LFS.
- Install and fetch tracked data with:
  ```bash
git lfs install
git lfs pull
  ```
- Additional local data files may be kept under `data/EURUSD/`, but do not commit private or transient datasets unless they are meant to be shared.

## Repository structure

- `include/`, `src/` — core C++ implementation
- `tests/` — unit tests and smoke tests
- `experiments/`, `benchmarks/` — research code and evaluation scripts
- `data/` — input datasets
- `results/` — generated experiment outputs

## Workflow

- Keep changes focused and small.
- Use `git status` and `git diff` before committing.
- Prefer `git add -p` to stage logical hunks.
- Write clear commit messages describing why a change was made.

## Style

- Follow the existing C++ code style in `src/` and `include/`.
- Prefer explicit directory structures rather than placing new files at the repository root.
