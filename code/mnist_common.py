"""Shared data and client operations for the original MNIST MLP configuration."""

from pathlib import Path

from filelock import FileLock
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def seed_generator(*parts):
    seed = int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0])
    return torch.Generator().manual_seed(seed)


def tensor_bytes(values):
    return sum(t.numel() * t.element_size() for t in values)


def load_mnist(data_root):
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(data_root / ".download.lock")):
        return MNIST(data_root, train=True, download=True), MNIST(data_root, train=False, download=True)


def partition(labels, seed):
    rng = np.random.RandomState(seed)
    clients = [[] for _ in range(50)]
    for label in range(10):
        indices = np.flatnonzero(labels.numpy() == label)
        rng.shuffle(indices)
        weights = rng.dirichlet([0.1] * len(clients))
        ends = (np.cumsum(weights) * len(indices)).astype(int)[:-1]
        for client, shard in zip(clients, np.split(indices, ends)):
            client.extend(shard)
    for client in clients:
        rng.shuffle(client)
    return [torch.tensor(x, dtype=torch.long) for x in clients if x]


def configure_fp32():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def normalize(images):
    return (images.float().div_(255).sub_(0.1307)).div_(0.3081)


def batches(dataset, ids, seed, round_id, client_id, device):
    order = ids[torch.randperm(len(ids), generator=seed_generator(seed, 1, round_id, client_id))]
    for subset in order.split(32):
        yield normalize(dataset.data[subset].to(device)), dataset.targets[subset].to(device)


def coordinate_bases(params, rank, seed, round_id, device):
    generator = seed_generator(seed, 2, round_id)
    bases = {}
    for name, weight in params.items():
        if weight.ndim != 2:
            continue
        width = weight.shape[1]
        retained = min(width, rank)
        indices = torch.randperm(width, generator=generator)[:retained]
        signs = torch.randint(2, (retained,), generator=generator).mul_(2).sub_(1)
        p = torch.zeros(width, retained)
        p[indices, torch.arange(retained)] = signs.float() * (width / retained) ** 0.5
        bases[name] = p.to(device)
    return bases


def client_momentum(params, bases):
    return {name: p.new_zeros((p.shape[0], bases[name].shape[1])) if name in bases
            else torch.zeros_like(p) for name, p in params.items()}


@torch.no_grad()
def encode_delta(params, initial, bases):
    payload = {}
    for name, p in params.items():
        delta = p - initial[name]
        payload[name] = delta if name not in bases else delta @ bases[name]
    return payload
