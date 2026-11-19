"""Pair FedSLoP, FedSub-CD, FedMuon and FedAvg-M on the original MNIST MLP."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fedmuon_matched import FedMuon, MUON_METHODS
from mnist_common import (
    SimpleMLP, batches, client_momentum, configure_fp32, encode_delta,
    input_bases, load_mnist, momentum_step, normalize, partition, tensor_bytes,
)

METHODS = ("FedSLoP", "FedAvg-M", "FedSub-CD", *MUON_METHODS)


@torch.no_grad()
def sub_step(local, initial, basis, coords, dual, lr, steps):
    for name, p in local.items():
        if name not in basis:
            p.add_(p.grad, alpha=-lr)
            continue
        project = basis[name]
        coords[name].add_(p.grad @ project, alpha=-lr * project.shape[1] / project.shape[0])
        if name in dual:
            coords[name].add_(dual[name], alpha=-1.0 / steps)
        p.copy_(initial[name]).add_(coords[name] @ project.T)


@torch.no_grad()
def update_duals(states, updates, average, previous_bases, new_bases):
    for client_id, payload in updates.items():
        state = states[client_id]
        for name in previous_bases:
            value = payload[name] - average[name]
            if name in state:
                value.add_(state[name])
            state[name] = value
    if new_bases is not previous_bases:
        rotations = {name: previous_bases[name].T @ new_bases[name] for name in previous_bases}
        for state in states:
            for name, value in state.items():
                state[name] = value @ rotations[name]


@torch.no_grad()
def evaluate(model, dataset, ids, device):
    correct = 0
    for subset in ids.split(256):
        x = normalize(dataset.data[subset].to(device))
        correct += model(x).argmax(dim=1).eq(dataset.targets[subset].to(device)).sum().item()
    return correct / len(ids)


def run(args):
    configure_fp32()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train, test = load_mnist(args.data_root)
    clients, validation_ids = partition(train.targets, args.seed, validation=args.tune)
    if len(clients) != 50:
        raise ValueError(f"Expected 50 nonempty clients, obtained {len(clients)}")
    server, model = SimpleMLP().to(device), SimpleMLP().to(device)
    initial, local = dict(server.named_parameters()), dict(model.named_parameters())
    full_bytes = tensor_bytes(initial.values())
    states = [{} for _ in clients]
    global_momentum = {name: torch.zeros_like(p) for name, p in initial.items()} if args.method == "FedAvg-M" else {}
    muon = None
    if args.method in MUON_METHODS:
        muon = FedMuon(initial, coefficient=0.9, compress_momentum=args.method == "FedMuon-SVD",
                       svd_rank_rule=args.svd_rank_rule, svd_fraction=args.svd_fraction,
                       uplink_budget=args.svd_uplink_budget)
    subspace = args.method in ("FedSLoP", "FedSub-CD")

    def get_basis(round_id):
        return input_bases(initial, 112, args.seed, round_id, device, coordinate=args.method == "FedSub-CD") if subspace else {}

    result = args.output
    result.mkdir(parents=True, exist_ok=False)
    (result / "config.json").write_text(json.dumps({
        **vars(args), "batch_size": 32, "local_epochs": 1, "client_count": 50,
        "dirichlet": 0.1, "projection_rank": 112 if subspace else None,
        "fedsub_refresh": 50 if args.method == "FedSub-CD" else None,
        "fedmuon_momentum_svd_ranks": muon.momentum_ranks if muon else None,
        "fedmuon_delta_svd_ranks": muon.delta_ranks if muon else None,
        "momentum": 0.9, "client_sizes": [len(c) for c in clients],
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }, indent=2, default=str) + "\n")

    basis = get_basis(0)
    cumulative_up = cumulative_down = 0
    with (result / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["round", "split", "accuracy", "uplink_bytes", "downlink_bytes",
                         "cumulative_up", "cumulative_down"])
        for t in range(args.rounds):
            if args.method == "FedSLoP":
                basis = get_basis(t)
            selected = np.random.RandomState(args.seed + t).choice(len(clients), args.active_clients, replace=False)
            size = len(selected)
            uplink = 0
            downlink = (full_bytes + (8 if basis else 0) + (muon.downlink_bytes() if muon else 0)) * size
            if muon:
                muon.start_round()
            aggregate, sub_updates = {}, {}
            for cid in selected:
                model.load_state_dict(server.state_dict())
                amps = client_momentum(local, basis) if args.method == "FedSLoP" else {}
                coords = client_momentum({name: local[name] for name in basis}, basis) if args.method == "FedSub-CD" else {}
                local_muon = muon.initial_state() if muon else None
                local_steps = math.ceil(len(clients[cid]) / 32)
                for x, y in batches(train, clients[cid], args.seed, t, int(cid), device):
                    model.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x), y)
                    loss.backward()
                    if args.method == "FedSub-CD":
                        sub_step(local, initial, basis, coords, states[cid], args.lr, local_steps)
                    elif muon:
                        muon.step(local, local_muon, args.lr)
                    else:
                        momentum_step(local, basis, amps, args.lr)
                if args.method == "FedSub-CD":
                    payload = {name: coords[name] if name in coords else p.detach() - initial[name].detach()
                               for name, p in local.items()}
                    sub_updates[cid] = coords
                else:
                    payload = encode_delta(local, initial, basis)
                if muon:
                    payload, sent_bytes = muon.transmit_model_delta(payload)
                    uplink += sent_bytes
                    uplink += muon.accept(payload, local_muon, local_steps, args.lr)
                else:
                    uplink += tensor_bytes(payload.values())
                for name, value in payload.items():
                    if name in aggregate:
                        aggregate[name].add_(value, alpha=1/size)
                    else:
                        aggregate[name] = value.mul(1/size)
            with torch.no_grad():
                if args.method == "FedSub-CD":
                    next_basis = get_basis(t + 1) if (t + 1) % 50 == 0 else basis
                    update_duals(states, sub_updates, aggregate, basis, next_basis)
                    downlink += size * tensor_bytes(aggregate[name] for name in basis)
                for name, value in aggregate.items():
                    update = value @ basis[name].T if name in basis else value
                    if global_momentum:
                        update = global_momentum[name].mul_(0.9).add_(update)
                    initial[name].add_(update)
                if args.method == "FedSub-CD":
                    basis = next_basis
                if muon:
                    muon.finish_round(size)
            cumulative_up += uplink
            cumulative_down += downlink
            accuracy, split = "", ""
            if t == 0 or (t + 1) % 10 == 0 or t + 1 == args.rounds:
                source, ids = (train, validation_ids) if args.tune else (test, torch.arange(len(test)))
                accuracy = evaluate(server, source, ids, device)
                split = "validation" if args.tune else "test"
                if not all(torch.isfinite(p).all().item() for p in initial.values()):
                    raise FloatingPointError("Nonfinite model")
                print(f"{args.method} seed={args.seed} round={t+1} {split}_accuracy={accuracy:.4f} up={uplink}", flush=True)
            writer.writerow([t+1, split, accuracy, uplink, downlink, cumulative_up, cumulative_down])
            file.flush()
    (result / "done").touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--active-clients", type=int, default=10, choices=(10,50))
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--svd-rank-rule", choices=("payload", "matrix"), default="payload")
    parser.add_argument("--svd-fraction", type=float, default=0.05)
    parser.add_argument("--svd-uplink-budget", type=int, help="Maximum bytes per client and round for both SVD messages.")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.svd_fraction <= 1:
        parser.error("--svd-fraction must be in (0, 1]")
    if args.svd_uplink_budget is not None and (args.svd_uplink_budget <= 0 or args.method != "FedMuon-SVD"):
        parser.error("--svd-uplink-budget requires a positive byte budget and --method FedMuon-SVD")
    run(args)


if __name__ == "__main__":
    main()
