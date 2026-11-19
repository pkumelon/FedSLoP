"""Plot the archived CIFAR-10 comparison against rounds and uplink bytes."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = (
    ("FedSLoP", "cifar10_formal_steps20_server_momentum/FedSLoP-rank75-server", "#255A9F", "-", "o"),
    ("FedAvg-M", "cifar10_formal_steps20_server_momentum/FedAvg-M", "#41464F", "--", "s"),
    ("FedSub-CD", "cifar10_formal_steps20_server_momentum/FedSub-CD", "#BD5E3E", "-.", "^"),
    ("FedLoRA-M", "cifar10_formal_steps20_server_momentum/FedLoRA-M", "#2A7E7D", ":", "D"),
    (r"FedMuon-SVD$^\dagger$", "cifar10_fedmuon_svd_budget/FedMuon-SVD", "#7654A4", "--", "v"),
)


def read_curves(results_root, relative):
    values, uplinks = [], []
    rounds = None
    for seed in (0, 1, 2):
        case = results_root / relative / f"seed-{seed}"
        if not (case / "done").exists():
            raise ValueError(f"Incomplete run: {case}")
        with (case / "metrics.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["evaluation_split"] == "test"]
        current = np.array([int(row["round"]) for row in rows])
        if not np.array_equal(current, np.arange(10, 201, 10)):
            raise ValueError(f"Unexpected evaluation rounds: {case}")
        rounds = current
        values.append([100 * float(row["accuracy"]) for row in rows])
        uplinks.append([int(row["cumulative_uplink_bytes"]) / 1e9 for row in rows])
    if any(not np.array_equal(uplinks[0], up) for up in uplinks[1:]):
        raise ValueError(f"Communication differs across seeds: {relative}")
    return rounds, np.array(uplinks[0]), np.mean(values, axis=0), np.std(values, axis=0, ddof=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("data/revision"))
    parser.add_argument("--output", type=Path, default=Path("picture/cifar10_accuracy_communication.pdf"))
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8.5, "axes.labelcolor": "#17447F",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7B8496", "axes.linewidth": .6,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.5, 2.7), sharey=True)
    for label, relative, color, linestyle, marker in ARMS:
        rounds, uplink, mean, sd = read_curves(args.results_root, relative)
        for axis, x in zip(axes, (rounds, uplink)):
            axis.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.55,
                      marker=marker, markevery=4, markersize=2.9)
            axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=.09, linewidth=0)
        print(f"{label}: {mean[-1]:.2f} +/- {sd[-1]:.2f}%, uplink {uplink[-1]:.6f} GB")
    axes[0].set_xlabel("(a) Communication rounds")
    axes[1].set_xlabel("(b) Cumulative uplink (GB)")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_xlim(0, 205)
    axes[1].set_xlim(0, 2.22)
    axes[0].set_xticks((0, 50, 100, 150, 200))
    axes[1].set_xticks((0, .5, 1, 1.5, 2))
    for axis in axes:
        axis.set_ylim(28, 90)
        axis.set_yticks((30, 40, 50, 60, 70, 80, 90))
        axis.grid(axis="y", color="#E5E8EE", linewidth=.6)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=7.4,
                  columnspacing=1.25, handlelength=2.5, bbox_to_anchor=(.52, 1.02))
    figure.subplots_adjust(top=.79, left=.07, right=.995, bottom=.19, wspace=.17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=.03)
    plt.close(figure)


if __name__ == "__main__":
    main()
