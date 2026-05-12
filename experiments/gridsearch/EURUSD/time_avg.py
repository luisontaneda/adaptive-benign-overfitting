import subprocess
import math
from typing import Dict, List

def parse_result_line(stdout: str) -> Dict[str, float]:
    # expects a line starting with: "RESULT key val key val ..."
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            toks = line.split()
            d: Dict[str, float] = {}
            i = 1
            while i + 1 < len(toks):
                d[toks[i]] = float(toks[i + 1])
                i += 2
            return d
    raise RuntimeError("No RESULT line found in stdout")

def mean_std(xs: List[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v)

def run_many(bin_path: str, args: List[str], n_runs: int) -> Dict[str, Dict[str, float]]:
    samples: Dict[str, List[float]] = {}
    for _ in range(n_runs):
        out = subprocess.check_output([bin_path, *args], text=True, stderr=subprocess.PIPE)
        res = parse_result_line(out)
        for k, v in res.items():
            samples.setdefault(k, []).append(v)

    summary: Dict[str, Dict[str, float]] = {}
    for k, xs in samples.items():
        m, s = mean_std(xs)
        summary[k] = {"mean": m, "std": s, "n": len(xs)}
    return summary

args = [
        "--run",
        "abo,qrd,krls",
        "--first_date",
        "7680",
        "--start_k",
        "0",
        "--end_k",
        "5",
        "--val_length",
        "1920",
        "--warmup",
        "50",
        "--abo_lags",
        "20",
        "--abo_window",
        "21",
        "--abo_sigma",
        "8.0",
        "--abo_log2D",
        "13",
        "--qrd_lags",
        "20",
        "--qrd_window",
        "272",
        "--krls_lags",
        "20",
        "--krls_window",
        "421",
        "--krls_sigma",
        "3.1"
]

stats = run_many("./bin/gridsearch_eurusd_test_best", args, n_runs=30)
for k, v in stats.items():
    print(k, v)
