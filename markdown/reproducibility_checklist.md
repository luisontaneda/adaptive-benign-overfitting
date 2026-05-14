# Reproducibility Checklist

## Before Running Benchmarks

- [ ] **System state stable**: CPU frequency scaling disabled, turbo boost off
- [ ] **No background processes**: Close other applications, minimize system load
- [ ] **Consistent environment**: Run on same machine, same OS version if possible
- [ ] **Data ready**: Ensure CSV data files are present and unchanged
- [ ] **Google Benchmark installed**: `libs/benchmark/` exists and is built

## Running Benchmarks

- [ ] **Document system info**:

  ```bash
  lscpu | grep "Model name"
  cat /proc/cpuinfo | grep "cpu MHz" | head -1
  uname -r
  g++ --version
  ```

- [ ] **Use consistent parameters**: Same hyperparameters (lags, window, sigma) for comparison

- [ ] **Run multiple times**: Benchmarks are stochastic; run at least 3 times on same system:

  ```bash
  for i in {1..3}; do
    ./bin/gridsearch_test_best_bench [args] \
      --benchmark_out=run_$i.json --benchmark_out_format=json
  done
  ```

- [ ] **Increase minimum time for stability**: Use `--benchmark_min_time=5.0` or higher for longer folds

- [ ] **Save outputs for later analysis**:
  ```bash
  --benchmark_out=results.json --benchmark_out_format=json
  ```

## Comparing Results

- [ ] **Use same dataset version**: Ensure data files haven't changed
- [ ] **Report both CPU and wall-clock time**: CPU time is more stable
- [ ] **Include error bars/variance**: Google Benchmark reports min/max/mean automatically
- [ ] **Save intermediate results**: Don't rely on console output; save JSON/CSV

## Sharing Results

Document these items in your report:

### Hardware

```
CPU: [Model name from lscpu]
Cores: [Number of cores]
Frequency: [CPU MHz]
Memory: [GB]
Cache: [L3 cache size]
```

### Software

```
Compiler: [GCC/Clang version]
Compiler flags: [From make/common.mk]
OS: [Linux distro + kernel version]
Build type: [Debug/Release]
```

### Benchmark Configuration

```
Command: [Full benchmark command with all parameters]
Data source: [Electricity / EUR-USD with dates/length]
Hyperparameters: [L, W, sigma, D for each model]
Warmup iterations: [50 or custom]
Minimum benchmark time: [Default 2.0s or custom]
```

### Results

```
Model: [ABO / QRD / KRLS]
Samples: [Number of iterations Google Benchmark ran]
Mean time (CPU): [XX.X ms]
Mean time (real): [XX.X ms]
Std deviation: [XX.X ms]
Min/Max: [min ms - max ms]
```

## Reproducibility Tips

1. **Use `taskset` for CPU affinity**:

   ```bash
   taskset -c 0-3 ./bin/gridsearch_test_best_bench ...
   ```

2. **Isolate cores on Linux**:

   ```bash
   sudo bash -c 'echo 1 > /proc/sys/kernel/sched_migration_cost_ns'
   sudo cpuset-isolate --isolate 2-3
   ```

3. **Disable address space layout randomization (ASLR)**:

   ```bash
   sudo bash -c 'echo 0 > /proc/sys/kernel/randomize_va_space'
   ```

4. **Compile with reproducible flags** (if using custom compiler):

   ```bash
   # Add to CXXFLAGS
   -fno-inline-small-functions
   -fno-vectorize
   ```

5. **Use `perf` for detailed profiling** (Linux):
   ```bash
   perf stat ./bin/gridsearch_test_best_bench ...
   ```

## Troubleshooting High Variance

| Symptom                     | Likely Cause                      | Solution                                                 |
| --------------------------- | --------------------------------- | -------------------------------------------------------- |
| 20%+ variance in results    | System noise                      | Increase `--benchmark_min_time=5.0`, disable CPU scaling |
| Results vary across runs    | Different data or hyperparameters | Verify parameters unchanged, check data checksums        |
| Benchmarks crash midway     | Memory pressure                   | Reduce `--val_length`, check available RAM               |
| Results too slow to measure | Cold cache                        | Let system stabilize; run without timing first           |
| CPU time >> real time       | Other processes                   | Use `taskset`, check for background tasks                |

## Example: Reproducible Benchmark Run

```bash
#!/bin/bash
set -e

# Log system info
{
  echo "=== REPRODUCIBILITY REPORT ==="
  echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "=== HARDWARE ==="
  lscpu | grep -E "Model name|CPU\(s\)|L3 cache"
  cat /proc/cpuinfo | grep "cpu MHz" | head -1
  echo ""
  echo "=== SOFTWARE ==="
  uname -a
  g++ --version | head -1
  echo ""
  echo "=== COMPILER FLAGS ==="
  grep "^CXXFLAGS" make/common.mk
  echo ""
  echo "=== SYSTEM CONFIGURATION ==="
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  cat /sys/devices/system/cpu/intel_pstate/no_turbo
  echo ""
  echo "=== BENCHMARK RUNS ==="
} | tee benchmark_report.txt

# Run benchmark 3 times
for run in 1 2 3; do
  echo "Run $run..." >> benchmark_report.txt
  ./bin/gridsearch_test_best_bench \
    --run abo,qrd,krls \
    --first_date 5376 --start_k 0 --end_k 5 --val_length 1344 --warmup 50 \
    --abo_lags 19 --abo_window 20 --abo_sigma 6.50586 --abo_log2D 11 \
    --qrd_lags 48 --qrd_window 128 \
    --krls_lags 25 --krls_window 261 --krls_sigma 4.2 \
    --benchmark_out="run_${run}.json" --benchmark_out_format=json \
    >> benchmark_report.txt 2>&1
done

echo "" >> benchmark_report.txt
echo "=== COMPLETED ===" >> benchmark_report.txt
echo "Results saved to: run_1.json, run_2.json, run_3.json"
cat benchmark_report.txt
```

Save this script and run it before committing performance-critical code.
