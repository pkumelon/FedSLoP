"""Measure one complete client update on the original FP32 MNIST MLP."""

import argparse
import json

import torch
from torch.nn import functional as F

from mnist_common import (
    SimpleMLP, batches, client_momentum, configure_fp32, encode_delta,
    input_bases, load_mnist, momentum_step, partition, tensor_bytes,
)


def measure(train, ids, seed, method, device):
    torch.manual_seed(seed)
    initial = SimpleMLP()
    model = SimpleMLP()
    model.load_state_dict(initial.state_dict())
    client = seed % len(ids)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    model.to(device)
    params = dict(model.named_parameters())
    snapshot = {name: p.detach().to(device) for name, p in initial.named_parameters()}
    bases = input_bases(params, 112, seed, 0, device) if method == "FedSLoP" else {}
    momentum = client_momentum(params, bases) if method != "FedAvg-M" else {}
    steps = 0
    for x, y in batches(train, ids[client], seed, 0, client, device):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        momentum_step(params, bases, momentum, 0.01839685275139246)
        steps += 1
    payload = encode_delta(params, snapshot, bases)
    torch.cuda.synchronize()
    return {
        "client_id": client,
        "local_steps": steps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
        "momentum_bytes": tensor_bytes(momentum.values()),
        "basis_bytes": tensor_bytes(bases.values()),
        "uplink_bytes": tensor_bytes(payload.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--method", choices=("FedSLoP", "FedAvg-M", "Full-client-momentum"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    configure_fp32()
    train, _ = load_mnist(args.data_root)
    clients, _ = partition(train.targets, args.seed)
    device = torch.device("cuda")
    measure(train, clients, args.seed, args.method, device)
    result = measure(train, clients, args.seed, args.method, device)
    print(json.dumps({
        "method": args.method, "seed": args.seed, "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "batch_size": 32, "local_epochs": 1, **result,
    }))


if __name__ == "__main__":
    main()
