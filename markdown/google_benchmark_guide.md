# Reproducible Benchmarking Guide

## Overview

This project now uses **Google Benchmark** for rigorous, reproducible performance measurements. Two new benchmark executables have been added:

- `gridsearch_test_best_bench` (electricity dataset)
- `gridsearch_eurusd_test_best_bench` (EUR/USD dataset)

The original CSV-output versions remain unchanged for backward compatibility:

- `gridsearch_test_best`
- `gridsearch_eurusd_test_best`

## Building Benchmark Executables

### Prerequisites

Google Benchmark must be available at `libs/benchmark/`:

```bash
# If not already present, clone Google Benchmark
git clone https://github.com/google/benchmark.git libs/benchmark
cd libs/benchmark
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
make -j$(nproc)
cd ../../..
```

### Build

```bash
make gridsearch_test_best_bench
make gridsearch_eurusd_test_best_bench

# Or build all benchmarks
make -j
```

## Running Benchmarks

### Basic Usage

```bash
# Run with default settings (electricity)
./bin/gridsearch_test_best_bench \
  --run abo,qrd,krls \
  --first_date 5376 \
  --start_k 0 \
  --end_k 5 \
  --val_length 1344 \
  --warmup 50 \
  --abo_lags 19 \
  --abo_window 20 \
  --abo_sigma 6.50586 \
  --abo_log2D 11 \
  --qrd_lags 48 \
  --qrd_window 128 \
  --krls_lags 25 \
  --krls_window 261 \
  --krls_sigma 4.2

# EUR/USD version
./bin/gridsearch_eurusd_test_best_bench \
  --run abo,qrd,krls \
  --first_date 7680 \
  --start_k 0 \
  --end_k 5 \
  --val_length 1920 \
  --warmup 50 \
  --abo_lags 19 \
  --abo_window 20 \
  --abo_sigma 6.50586 \
  --abo_log2D 11 \
  --qrd_lags 48 \
  --qrd_window 128 \
  --krls_lags 25 \
  --krls_window 261 \
  --krls_sigma 4.2
```

## Google Benchmark Options

By default, the benchmarks are configured with:

- `--benchmark_min_time=2.0` — minimum time per benchmark (ensures sufficient samples)
- `--benchmark_iterations=1` — run each full fold once per iteration
- `--benchmark_measure_process_cpu_time` — measure CPU time
- `--benchmark_use_real_time` — also report real (wall-clock) time

### Additional Benchmark Flags

```bash
# List all available benchmarks
./bin/gridsearch_test_best_bench --benchmark_list_tests

# Run only ABO benchmark
./bin/gridsearch_test_best_bench --benchmark_filter=BM_ABO_Fold

# Set output format (default: console, alternatives: csv, json)
./bin/gridsearch_test_best_bench --benchmark_out=results.json --benchmark_out_format=json

# Verbose output with statistics
./bin/gridsearch_test_best_bench --benchmark_verbosity=2

# Save results to CSV for comparison
./bin/gridsearch_test_best_bench --benchmark_out=bench_results.csv --benchmark_out_format=csv

# Increase minimum time for more stable results
./bin/gridsearch_test_best_bench --benchmark_min_time=5.0
```

## Reproducibility Best Practices

### 1. System Configuration

Before benchmarking, stabilize your system:

```bash
# Disable CPU frequency scaling (Linux, requires root)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable turbo boost for consistent performance
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Set CPU affinity (bind to specific cores)
taskset -c 0-3 ./bin/gridsearch_test_best_bench ...
```

### 2. Warmup and Repetition

- The `--warmup` parameter (50 by default) skips initial iterations to reach steady state
- Google Benchmark automatically runs multiple iterations for statistical confidence
- Always run benchmarks multiple times on the same system to capture variance

### 3. Environment Logging

To reproduce exact conditions, document:

- CPU model: `lscpu | grep "Model name"`
- CPU frequency: `cat /proc/cpuinfo | grep "cpu MHz"`
- Kernel version: `uname -r`
- Compiler: `g++ --version`
- Build flags: Check `make/common.mk`

Example:

```bash
# Create a benchmark report with system info
{
  echo "=== System Info ==="
  lscpu | grep "Model name"
  cat /proc/cpuinfo | grep "cpu MHz" | head -1
  uname -r
  g++ --version | head -1

  echo ""
  echo "=== Benchmark Results ==="
  ./bin/gridsearch_test_best_bench \
    --run abo,qrd,krls \
    --first_date 5376 --start_k 0 --end_k 5 --val_length 1344 --warmup 50 \
    --abo_lags 19 --abo_window 20 --abo_sigma 6.50586 --abo_log2D 11 \
    --qrd_lags 48 --qrd_window 128 \
    --krls_lags 25 --krls_window 261 --krls_sigma 4.2 \
    --benchmark_out=bench_report.json --benchmark_out_format=json
} | tee benchmark_run.log
```

### 4. Comparison Across Versions

Save benchmark outputs for comparison:

```bash
# Baseline (e.g., current main)
./bin/gridsearch_test_best_bench [...args...] \
  --benchmark_out=baseline.json --benchmark_out_format=json

# After optimization
./bin/gridsearch_test_best_bench [...args...] \
  --benchmark_out=optimized.json --benchmark_out_format=json

# Diff (Google Benchmark tools can compare JSON outputs)
# See: https://github.com/google/benchmark#comparing-results
```

## Output Interpretation

### Console Output Example

```
Benchmark                Time             CPU       Iterations   Label
────────────────────────────────────────────────────────────────────────
BM_ABO_Fold         3457 ms         3421 ms            1     L=19,W=20,sigma=6.50586,D=2048
BM_QRD_Fold         1234 ms         1210 ms            1     L=48,W=128
BM_KRLS_Fold         987 ms          965 ms            1     L=25,W=261,sigma=4.2
```

- **Time**: Wall-clock time (affected by system load)
- **CPU**: Actual CPU time used (more stable)
- **Iterations**: Number of times the benchmark was run
- **Label**: Model hyperparameters for easy identification

### CSV Output

When using `--benchmark_out=results.csv --benchmark_out_format=csv`, you get tabular data suitable for Excel/Python analysis.

### JSON Output

JSON format is ideal for scripting and automated comparison:

```bash
./bin/gridsearch_test_best_bench ... --benchmark_out=results.json --benchmark_out_format=json
python3 scripts/compare_benchmarks.py baseline.json optimized.json
```

## Troubleshooting

### Benchmark Not Found

```
Error: Skipping gridsearch_test_best_bench: Google Benchmark not found (libs/benchmark/)
```

**Solution**: Install Google Benchmark (see Prerequisites above)

### Variable Results

- **Cause**: System noise, CPU frequency scaling, other processes running
- **Solution**: Use `taskset`, disable CPU scaling, run in quiet environment, increase `--benchmark_min_time`

### Memory Usage

- **Issue**: Benchmarks consume significant memory due to data loading
- **Solution**: Reduce `--val_length` for quick iterations, or run on a system with sufficient RAM

## Migration from Old Timing Code

The old manual timing (using `std::chrono`) is still available in the non-benchmark executables (`gridsearch_test_best`). The benchmark versions provide:

- ✓ Automatic statistical aggregation
- ✓ Built-in warmup and outlier rejection
- ✓ Standard JSON/CSV export for reproducibility
- ✓ Better compiler optimization handling via `DoNotOptimize()`
- ✓ CPU vs. wall-clock time separation

To migrate existing scripts:

1. Replace calls to `bin/gridsearch_test_best` with `bin/gridsearch_test_best_bench`
2. Parse JSON output instead of CSV (samples are pre-aggregated by Google Benchmark)
3. Use `--benchmark_out_format=csv` if CSV is required, but note that Google Benchmark's CSV format differs from the original

## See Also

- [Google Benchmark Documentation](https://github.com/google/benchmark/tree/main/docs)
- [Benchmarking Best Practices](https://easyperf.net/blog/2019/08/02/Perf-measurement-checklist/)
- `make/benchmarks.mk` — Benchmark build configuration
- `experiments/gridsearch/electricity/test_elect_best_hyperpar_bench.cpp` — Benchmark source code
