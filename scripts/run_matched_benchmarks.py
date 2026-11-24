"""Run the final MNIST and CIFAR-10 benchmarks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
from statistics import mean, stdev
import subprocess
import sys

from filelock import FileLock

MNIST_LR = 0.01839685275139246
CIFAR_GRID = ("0.015", "0.03", "0.06", "0.09", "0.12")
METHODS = {
    "mnist": ("FedSub-CD", "FedMuon", "FedMuon-SVD"),
    "cifar": ("FedSLoP", "FedAvg-M", "FedSub-CD", "FedLoRA-M"),
}
CODE = Path(__file__).resolve().parents[1] / "code"


def case_command(args, method, seed, lr, path, tuning):
    if args.dataset == "mnist":
        script = CODE / "mnist_matched.py"
        flags = ["--rounds", "100"]
    else:
        script = CODE / "cifar10_resnet20.py"
        rank = "0.75" if method in ("FedSLoP", "FedSub-CD", "FedLoRA-M") else "1"
        flags = [
            "--rounds", "30" if tuning else "200", "--local-steps", "20", "--batch-size", "64",
            "--clients", "10", "--alpha", "0.3", "--weight-decay", "0.0001",
            "--client-momentum", "0.8", "--server-momentum", "0.8",
            "--server-stepsize", "1", "--rank-ratio", rank, "--eval-every", "10",
        ]
        if method == "FedSLoP":
            flags.extend(["--fedslop-server-momentum", "--inplace-projected-step"])
        if tuning:
            flags.append("--tuning")
    return [
        sys.executable, "-u", str(script), "--method", method, "--seed", str(seed), "--lr", lr,
        "--output", str(path), "--data-root", str(args.data_root), "--device", args.device, *flags,
    ]


def read_result(path):
    with (path / "metrics.csv").open() as f:
        return list(csv.DictReader(f))[-1]


def run_case(args, method, seed, lr, path, tuning, *, cuda_device=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path / "done").exists():
        cfg = json.loads((path / "config.json").read_text())
        assert cfg["method"] == method and cfg["seed"] == seed and cfg["lr"] == float(lr), cfg
        return read_result(path)
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path.rename(path.with_name(path.name + f".incomplete-{stamp}"))
    print(f"{'screen' if tuning else 'formal'} {args.dataset} {method} seed={seed} lr={lr}", flush=True)
    command = case_command(args, method, seed, lr, path, tuning)
    # Emit child progress live and preserve its error text for nonfinite trial handling.
    lines = []
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    if process.returncode:
        if tuning and any("FloatingPointError: Nonfinite" in line for line in lines):
            return None
        raise subprocess.CalledProcessError(process.returncode, command)
    row = read_result(path)
    assert row.get("split", row.get("evaluation_split")) == ("validation" if tuning else "test")
    return row


def run_cases(args, cases):
    if not args.cuda_devices:
        return [run_case(args, *spec) for spec in cases]

    pending = Queue()
    for index, spec in enumerate(cases):
        pending.put((index, spec))
    results = [None] * len(cases)

    def worker(device):
        while True:
            try:
                index, spec = pending.get_nowait()
            except Empty:
                return
            results[index] = run_case(args, *spec, cuda_device=device)

    with ThreadPoolExecutor(max_workers=len(args.cuda_devices)) as pool:
        futures = [pool.submit(worker, device) for device in args.cuda_devices]
        for future in futures:
            future.result()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mnist", "cifar"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--method", help="Run one method from the selected dataset's comparison.")
    p.add_argument("--device", choices=("cuda", "cpu"), help="Defaults to CPU for MNIST and CUDA for CIFAR-10.")
    p.add_argument("--cuda-devices", nargs="+", help="Schedule independent runs over these GPU indices.")
    p.add_argument("--tune-only", action="store_true", help="Run only CIFAR-10 learning-rate selection.")
    args = p.parse_args()
    if args.method and args.method not in METHODS[args.dataset]:
        p.error(f"--method for {args.dataset} must be one of: {', '.join(METHODS[args.dataset])}")
    if args.tune_only and args.dataset != "cifar":
        p.error("--tune-only requires --dataset cifar")
    if args.device is None:
        args.device = "cpu" if args.dataset == "mnist" else "cuda"
    if args.cuda_devices and args.device != "cuda":
        p.error("--cuda-devices requires --device cuda")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    args.output.mkdir(parents=True, exist_ok=True)
    # Serialize schedulers targeting the same output so independent jobs can safely resume it.
    with FileLock(str(args.output / ".runner.lock")):
        run_comparison(args)


def select_cifar_rates(args, arms):
    rates = {}
    for method in arms:
        cases = [(method, 0, lr, args.output / "tuning" / method / lr, True) for lr in CIFAR_GRID]
        scores = {lr: None if row is None else float(row["accuracy"])
                  for lr, row in zip(CIFAR_GRID, run_cases(args, cases))}
        valid = [lr for lr in CIFAR_GRID if scores[lr] is not None]
        if not valid:
            raise RuntimeError(f"All validation candidates diverged for {method}")
        chosen = max(valid, key=lambda lr: (scores[lr], -float(lr)))
        rates[method] = {"lr": chosen, "validation_scores": scores}
        print(f"selected {method} {chosen}", flush=True)
    (args.output / "selected_lr.json").write_text(json.dumps(rates, indent=2) + "\n")
    return {method: result["lr"] for method, result in rates.items()}


def run_comparison(args):
    arms = (args.method,) if args.method else METHODS[args.dataset]
    mnist = args.dataset == "mnist"
    rates = {method: str(MNIST_LR) for method in arms} if mnist else select_cifar_rates(args, arms)
    if args.tune_only:
        return

    summary = []
    for method, chosen in rates.items():
        cases = [(method, seed, chosen, args.output / method / f"seed-{seed}", False)
                 for seed in (0, 1, 2)]
        rows = run_cases(args, cases)
        values = [float(row["accuracy"]) for row in rows]
        if mnist:
            uplink_field = "uplink_bytes_per_client_round"
            payloads = [int(row["uplink_bytes"]) // 10 for row in rows]
        else:
            uplink_field = "total_uplink_bytes"
            payloads = [int(row["cumulative_uplink_bytes"]) for row in rows]
        if len(set(payloads)) != 1:
            raise ValueError(f"Inconsistent uplink totals for {method}: {payloads}")
        summary.append({"method": method, "lr": chosen, "accuracy_mean": mean(values),
                        "accuracy_sd": stdev(values), uplink_field: payloads[0]})
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
