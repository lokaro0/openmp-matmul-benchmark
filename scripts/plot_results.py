import csv
from pathlib import Path
from collections import defaultdict
from statistics import median
import math
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "raw_results.csv"

RUN_NAME = "run_02"
RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME

OUTPUT_PATH = PROJECT_ROOT / "results" / "summary.csv"
SUMMARY_COLUMNS = (
    "implementation",
    "matrix_size",
    "configured_threads",
    "measurement_count",
    "median_seconds",
    "min_seconds",
    "max_seconds",
    "median_gflops",
)

SPEEDUP_PATH = PROJECT_ROOT / "results" / "speedup.csv"
SPEEDUP_COLUMNS = (
    "matrix_size",
    "configured_threads",
    "order_speedup",
    "openmp_speedup",
    "total_speedup",
    "thread_scaling_speedup",
)

MEASUREMENT_COUNT = 5
IMPLEMENTATIONS = ("ijk_serial", "ikj_serial", "ikj_openmp")
MATRIX_SIZES = (256, 512, 1024)
THREAD_COUNTS = (1, 2, 4, 8)

def calculate_gflops (elapsed_time, n):
    flops = 2.0 * n * n * n
    flops_per_second = flops / elapsed_time / 1e9
    return flops_per_second


with INPUT_PATH.open(
    "r", newline="", encoding="utf-8"
) as input_file:
    reader = csv.DictReader(input_file)
    rows = list(reader)

groups = defaultdict(list)
trial_groups = defaultdict(list)

for row in rows:
    key = (
        row["implementation"],
        int(row["matrix_size"]),
        int(row["configured_threads"]),
    )

    seconds = float(row["seconds"])

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid measurement time: {seconds}")

    groups[key].append(seconds)
    trial_groups[key].append(int(row["trial"]))

expected_keys = {
    (implementation, matrix_size, thread_count)
    for implementation in IMPLEMENTATIONS
    for matrix_size in MATRIX_SIZES
    for thread_count in THREAD_COUNTS
}
actual_keys = set(groups)

missing_keys = expected_keys - actual_keys
unexpected_keys = actual_keys - expected_keys

if missing_keys or unexpected_keys:
    raise ValueError(f"Missing groups: {missing_keys}\n" f"unexpected groups: {unexpected_keys}")

summary_rows = []

median_times = {}
lower_errors = {}
upper_errors = {}

for key, seconds_values in groups.items():
    if sorted(trial_groups[key]) != list(range(1, MEASUREMENT_COUNT + 1)):
        raise ValueError(f"Invalid trials for {key}: {trial_groups[key]}")
    implementation, matrix_size, configured_threads = key

    median_seconds = median(seconds_values)
    min_seconds = min(seconds_values)
    max_seconds = max(seconds_values)
    median_gflops = calculate_gflops(median_seconds, matrix_size)

    median_times[key] = median_seconds
    lower_errors[key] = median_seconds - min_seconds
    upper_errors[key] = max_seconds - median_seconds

    summary_rows.append((*key, len(seconds_values), median_seconds, min_seconds, max_seconds, median_gflops))

speedups = {}
for matrix_size in MATRIX_SIZES:
    for configured_threads in THREAD_COUNTS:
        ijk_seconds = median_times["ijk_serial", matrix_size, configured_threads]
        ikj_seconds = median_times["ikj_serial", matrix_size, configured_threads]
        openmp_seconds = median_times["ikj_openmp", matrix_size, configured_threads]
        openmp_single_thread_seconds = median_times["ikj_openmp", matrix_size, 1]

        order_speedup = ijk_seconds / ikj_seconds
        openmp_speedup = ikj_seconds / openmp_seconds
        total_speedup = ijk_seconds / openmp_seconds
        thread_scaling_speedup = openmp_single_thread_seconds / openmp_seconds

        speedups[matrix_size, configured_threads] = {
            "order_speedup": order_speedup,
            "openmp_speedup": openmp_speedup,
            "total_speedup": total_speedup,
            "thread_scaling_speedup": thread_scaling_speedup,
        }

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as output_file:
    writer = csv.writer(output_file)
    writer.writerow(SUMMARY_COLUMNS)
    writer.writerows(summary_rows)

with SPEEDUP_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as speedup_file:
    writer = csv.writer(speedup_file)
    writer.writerow(SPEEDUP_COLUMNS)

    for key, values in speedups.items():
        writer.writerow((
            *key,
            values["order_speedup"],
            values["openmp_speedup"],
            values["total_speedup"],
            values["thread_scaling_speedup"],
        ))

for matrix_size in MATRIX_SIZES:
    fig, ax = plt.subplots()

    for implementation in IMPLEMENTATIONS:
        keys = [
            (implementation, matrix_size, thread_count)
            for thread_count in THREAD_COUNTS
        ]

        median_values = [median_times[key] for key in keys]
        lower_values = [lower_errors[key] for key in keys]
        upper_values = [upper_errors[key] for key in keys]

        ax.errorbar(
            THREAD_COUNTS,
            median_values,
            yerr=[lower_values, upper_values],
            marker="o",
            capsize=4,
            label=implementation
        )

    ax.set_title(f"Matrix multiplication: n={matrix_size}")
    ax.set_xlabel("Configured OpenMP threads")
    ax.set_ylabel("Elapsed time [s]")
    ax.set_yscale("log")
    ax.set_xticks(THREAD_COUNTS)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PROJECT_ROOT / "results" / f"execution_time_{matrix_size}.png",
        dpi=150
    )
    plt.close(fig)

fig, ax = plt.subplots()

for matrix_size in MATRIX_SIZES:
    openmp_values = [
        speedups[(matrix_size, thread_count)]["openmp_speedup"]
        for thread_count in THREAD_COUNTS
    ]

    ax.plot(
        THREAD_COUNTS,
        openmp_values,
        marker="o",
        label=f"n={matrix_size}"
    )

ax.axhline(
    1.0,
    color="gray",
    linestyle="--",
    label="No speedup"
)

ax.set_title("OpenMP speedup over ikj_serial")
ax.set_xlabel("Configured OpenMP threads")
ax.set_ylabel("Speedup [times]")
ax.set_xticks(THREAD_COUNTS)
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    PROJECT_ROOT / "results" / "openmp_speedup.png",
    dpi=150
)
plt.close(fig)

fig, ax = plt.subplots()

for matrix_size in MATRIX_SIZES:
    openmp_values = [
        speedups[(matrix_size, thread_count)]["thread_scaling_speedup"]
        for thread_count in THREAD_COUNTS
    ]

    ax.plot(
        THREAD_COUNTS,
        openmp_values,
        marker="o",
        label=f"n={matrix_size}"
    )

ax.axhline(
    1.0,
    color="gray",
    linestyle="--",
    label="No speedup"
)

ax.set_title("OpenMP scaling relative to 1 thread")
ax.set_xlabel("Configured OpenMP threads")
ax.set_ylabel("Speedup [times]")
ax.set_xticks(THREAD_COUNTS)
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    PROJECT_ROOT / "results" / "openmp_scaling.png",
    dpi=150
)
plt.close(fig)
