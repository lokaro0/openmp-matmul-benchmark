import csv
import os
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

MATRIX_SIZES = (256, 512, 1024)
THREAD_COUNTS = (1, 2, 4, 8)

CSV_COLUMNS = (
    "trial",
    "implementation",
    "matrix_size",
    "configured_threads",
    "seconds",
    "gflops",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_PATH = PROJECT_ROOT / "build" / "matmul_benchmark.exe"
OUTPUT_PATH = PROJECT_ROOT / "results" / "raw_results.csv"

environment = os.environ.copy()
environment["OMP_DYNAMIC"] = "FALSE"

result_rows = []
for matrix_size in MATRIX_SIZES:
    for thread_count in THREAD_COUNTS:
        environment["OMP_NUM_THREADS"] = str(thread_count)
        completed_process = subprocess.run(
            [str(EXECUTABLE_PATH), str(matrix_size)],
            capture_output=True,
            text=True,
            env=environment,
        )

        if completed_process.returncode != 0:
            print(completed_process.stderr)
            raise RuntimeError(
                f"Benchmark failed: "
                f"matrix_size={matrix_size}, "
                f"thread_count={thread_count}"
            )

        reader = csv.reader(completed_process.stdout.splitlines())
        result_rows.extend(reader)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as output_file:
    writer = csv.writer(output_file)
    writer.writerow(CSV_COLUMNS)
    writer.writerows(result_rows)
