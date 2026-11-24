"""Run the three additional MNIST baselines on the original MLP."""

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
    coordinate_bases, load_mnist, normalize, partition, tensor_bytes,
)

METHODS = ("FedSub-CD", *MUON_METHODS)
ACTIVE_CLIENTS = 10
SVD_UPLINK_BUDGET = 62376


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
    clients = partition(train.targets, args.seed)
    if len(clients) != 50:
        raise ValueError(f"Expected 50 nonempty clients, obtained {len(clients)}")
    server, model = SimpleMLP().to(device), SimpleMLP().to(device)
    initial, local = dict(server.named_parameters()), dict(model.named_parameters())
    full_bytes = tensor_bytes(initial.values())
    states = [{} for _ in clients]
    uplink_budget = SVD_UPLINK_BUDGET if args.method == "FedMuon-SVD" else None
    muon = FedMuon(initial, coefficient=0.9, uplink_budget=uplink_budget) if args.method in MUON_METHODS else None
    subspace = args.method == "FedSub-CD"

    result = args.output
    result.mkdir(parents=True, exist_ok=False)
    (result / "config.json").write_text(json.dumps({
        **vars(args), "batch_size": 32, "local_epochs": 1, "client_count": 50,
        "active_clients": ACTIVE_CLIENTS, "svd_uplink_budget": uplink_budget,
        "dirichlet": 0.1, "projection_rank": 112 if subspace else None,
        "fedsub_refresh": 50 if args.method == "FedSub-CD" else None,
        "fedmuon_momentum_svd_ranks": muon.momentum_ranks if muon else None,
        "fedmuon_delta_svd_ranks": muon.delta_ranks if muon else None,
        "momentum": 0.9, "client_sizes": [len(c) for c in clients],
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }, indent=2, default=str) + "\n")

    basis = coordinate_bases(initial, 112, args.seed, 0, device) if subspace else {}
    cumulative_up = cumulative_down = 0
    with (result / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["round", "split", "accuracy", "uplink_bytes", "downlink_bytes",
                         "cumulative_up", "cumulative_down"])
        for t in range(args.rounds):
            selected = np.random.RandomState(args.seed + t).choice(len(clients), ACTIVE_CLIENTS, replace=False)
            size = len(selected)
            uplink = 0
            downlink = (full_bytes + (8 if basis else 0) + (muon.downlink_bytes() if muon else 0)) * size
            if muon:
                muon.start_round()
            aggregate, sub_updates = {}, {}
            for cid in selected:
                model.load_state_dict(server.state_dict())
                coords = client_momentum({name: local[name] for name in basis}, basis)
                local_muon = muon.initial_state() if muon else None
                local_steps = math.ceil(len(clients[cid]) / 32)
                for x, y in batches(train, clients[cid], args.seed, t, int(cid), device):
                    model.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x), y)
                    loss.backward()
                    if subspace:
                        sub_step(local, initial, basis, coords, states[cid], args.lr, local_steps)
                    else:
                        muon.step(local, local_muon, args.lr)
                if subspace:
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
                if subspace:
                    next_basis = basis
                    if (t + 1) % 50 == 0:
                        next_basis = coordinate_bases(initial, 112, args.seed, t + 1, device)
                    update_duals(states, sub_updates, aggregate, basis, next_basis)
                    downlink += size * tensor_bytes(aggregate[name] for name in basis)
                for name, value in aggregate.items():
                    update = value @ basis[name].T if name in basis else value
                    initial[name].add_(update)
                if subspace:
                    basis = next_basis
                if muon:
                    muon.finish_round(size)
            cumulative_up += uplink
            cumulative_down += downlink
            accuracy, split = "", ""
            if t == 0 or (t + 1) % 10 == 0 or t + 1 == args.rounds:
                accuracy = evaluate(server, test, torch.arange(len(test)), device)
                split = "test"
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
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
