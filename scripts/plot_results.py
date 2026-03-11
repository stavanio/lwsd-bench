#!/usr/bin/env python3
"""
Plot LWSD experiment results.

Reads the CSV files from run_experiment.sh and generates
paper-ready figures.

Usage:
    python3 plot_results.py /path/to/lwsd_experiment_YYYYMMDD_HHMMSS

Generates:
    1. lwsd_timeseries.pdf   - LWSD over time for all three delays
    2. lwsd_boxplot.pdf      - Distribution comparison
    3. error_vs_latency.pdf  - State error vs injected latency
    4. summary_table.txt     - LaTeX table for the paper
"""

import sys
import os
import csv
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found. Install with: pip install matplotlib")
    print("Generating text summary only.\n")


def load_csv(path):
    """Load LWSD CSV, skipping comment lines."""
    data = {
        "timestamp": [],
        "lwsd": [],
        "lwsd_rate": [],
        "state_error_m": [],
        "inference_latency_ms": [],
        "staleness_mm": [],
    }

    with open(path, "r") as f:
        reader = csv.DictReader(
            (row for row in f if not row.startswith("#"))
        )
        for row in reader:
            try:
                for key in data:
                    data[key].append(float(row[key]))
            except (ValueError, KeyError):
                continue

    for key in data:
        data[key] = np.array(data[key])

    # Normalize timestamps to start at 0
    if len(data["timestamp"]) > 0:
        data["timestamp"] -= data["timestamp"][0]

    return data


def print_summary(results):
    """Print summary table to console and LaTeX."""
    print("=" * 65)
    print(f"  {'Delay':>8s}  {'Mean LWSD':>10s}  {'Max LWSD':>10s}  "
          f"{'Mean Err':>10s}  {'Max Err':>10s}")
    print(f"  {'(ms)':>8s}  {'':>10s}  {'':>10s}  "
          f"{'(mm)':>10s}  {'(mm)':>10s}")
    print("-" * 65)

    for delay, data in sorted(results.items()):
        if len(data["lwsd"]) == 0:
            continue
        print(
            f"  {delay:>8d}  "
            f"{np.mean(data['lwsd']):>10.4f}  "
            f"{np.max(data['lwsd']):>10.4f}  "
            f"{np.mean(data['staleness_mm']):>10.1f}  "
            f"{np.max(data['staleness_mm']):>10.1f}"
        )
    print("=" * 65)


def generate_latex_table(results, output_path):
    """Generate LaTeX table for the paper."""
    with open(output_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Measured LWSD on ROSMASTER M1 with injected latency. "
            "Same path, same environment dynamics, three latency conditions.}\n"
        )
        f.write("\\label{tab:measured_lwsd}\n")
        f.write("\\begin{tabular}{@{}rcccc@{}}\n")
        f.write("\\toprule\n")
        f.write(
            "\\textbf{Injected Delay (ms)} & "
            "\\textbf{Mean LWSD} & \\textbf{Max LWSD} & "
            "\\textbf{Mean Error (mm)} & \\textbf{Max Error (mm)} \\\\\n"
        )
        f.write("\\midrule\n")

        for delay, data in sorted(results.items()):
            if len(data["lwsd"]) == 0:
                continue
            f.write(
                f"{delay} & "
                f"{np.mean(data['lwsd']):.4f} & "
                f"{np.max(data['lwsd']):.4f} & "
                f"{np.mean(data['staleness_mm']):.1f} & "
                f"{np.max(data['staleness_mm']):.1f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to {output_path}")


def plot_timeseries(results, output_path):
    """Time series of LWSD for all delay conditions."""
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = {0: "#2196F3", 200: "#FF9800", 500: "#F44336"}
    labels = {0: "Native (~15 ms)", 200: "+200 ms", 500: "+500 ms"}

    for delay, data in sorted(results.items()):
        if len(data["lwsd"]) == 0:
            continue
        ax.plot(
            data["timestamp"],
            data["lwsd"],
            color=colors.get(delay, "gray"),
            label=labels.get(delay, f"{delay} ms"),
            linewidth=1.2,
            alpha=0.85,
        )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("LWSD", fontsize=11)
    ax.set_title("Latency-Weighted State Divergence Over Time", fontsize=12)
    ax.legend(title="Injected Latency", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Time series plot saved to {output_path}")


def plot_boxplot(results, output_path):
    """Box plot comparing LWSD distributions."""
    fig, ax = plt.subplots(figsize=(6, 4))

    data_list = []
    labels = []
    for delay in sorted(results.keys()):
        if len(results[delay]["lwsd"]) > 0:
            data_list.append(results[delay]["lwsd"])
            labels.append(f"{delay} ms")

    bp = ax.boxplot(data_list, labels=labels, patch_artist=True)

    colors = ["#2196F3", "#FF9800", "#F44336"]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel("Injected Latency", fontsize=11)
    ax.set_ylabel("LWSD", fontsize=11)
    ax.set_title("LWSD Distribution by Latency Condition", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Box plot saved to {output_path}")


def plot_error_vs_latency(results, output_path):
    """Scatter: measured latency vs state error, colored by condition."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {0: "#2196F3", 200: "#FF9800", 500: "#F44336"}
    labels = {0: "Native", 200: "+200 ms", 500: "+500 ms"}

    for delay, data in sorted(results.items()):
        if len(data["inference_latency_ms"]) == 0:
            continue
        ax.scatter(
            data["inference_latency_ms"],
            data["staleness_mm"],
            color=colors.get(delay, "gray"),
            label=labels.get(delay, f"{delay} ms"),
            alpha=0.4,
            s=8,
        )

    # Theoretical line: error = v * latency, for v = 0.3 m/s
    lat_range = np.linspace(0, 600, 100)
    ax.plot(
        lat_range,
        0.3 * lat_range,  # 0.3 m/s * ms = 0.3 mm/ms
        "k--",
        linewidth=1.5,
        label="Theoretical (v=0.3 m/s)",
        alpha=0.7,
    )

    ax.set_xlabel("Measured Inference Latency (ms)", fontsize=11)
    ax.set_ylabel("State Error (mm)", fontsize=11)
    ax.set_title("State Error vs. Perception Latency", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Error vs latency plot saved to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <experiment_directory>")
        print("  e.g.: python3 plot_results.py ~/lwsd_experiment_20260310_140000")
        sys.exit(1)

    exp_dir = sys.argv[1]

    # Load results
    results = {}
    for delay in [0, 200, 500]:
        csv_path = os.path.join(exp_dir, f"delay_{delay}ms", "lwsd_log.csv")
        if os.path.exists(csv_path):
            results[delay] = load_csv(csv_path)
            print(f"Loaded {len(results[delay]['lwsd'])} samples from {delay}ms trial")
        else:
            print(f"Warning: {csv_path} not found, skipping")

    if not results:
        print("No data found!")
        sys.exit(1)

    # Summary
    print_summary(results)

    # LaTeX table
    generate_latex_table(results, os.path.join(exp_dir, "summary_table.tex"))

    # Plots
    if HAS_MPL:
        plot_timeseries(results, os.path.join(exp_dir, "lwsd_timeseries.pdf"))
        plot_boxplot(results, os.path.join(exp_dir, "lwsd_boxplot.pdf"))
        plot_error_vs_latency(
            results, os.path.join(exp_dir, "error_vs_latency.pdf")
        )
    else:
        print("Skipping plots (matplotlib not installed)")


if __name__ == "__main__":
    main()
