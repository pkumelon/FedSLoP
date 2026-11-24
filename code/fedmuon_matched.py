"""FedMuon Algorithm 1 in the matched experiments.

Uses the authors' five-step Newton–Schulz coefficients for matrix
orthogonalization, with FP32 arithmetic and the benchmark's shared momentum.

Reference: https://arxiv.org/abs/2510.27403
FedMuon-SVD dagger compresses both momentum and model increments under a
shared uplink budget.
"""

import torch

# Default local–global alignment weight in the authors' main_FedMuon.py.
ALIGNMENT = 0.85
SVD_FRACTION = 0.05
MUON_METHODS = ("FedMuon", "FedMuon-SVD")


def orthogonalize(momentum):
    matrix = momentum.reshape(momentum.shape[0], -1)
    transpose = matrix.shape[0] > matrix.shape[1]
    x = matrix.T if transpose else matrix
    x = x / (x.norm() + 1e-7)
    for _ in range(5):
        gram = x @ x.T
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    return (x.T if transpose else x).reshape_as(momentum)


def svd_component_bytes(value):
    rows = value.shape[0]
    return (rows + value.numel() // rows + 1) * value.element_size()


def decode_truncated_svd(value, rank):
    matrix = value.reshape(value.shape[0], -1)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    u, singular_values, vh = u[:, :rank], singular_values[:rank], vh[:rank]
    size = sum(factor.numel() * factor.element_size() for factor in (u, singular_values, vh))
    return ((u * singular_values) @ vh).reshape_as(value), size


def model_delta_ranks(params, budget):
    """Use the remaining uplink budget for model deltas, including dense vectors."""
    matrices = {name: value for name, value in params.items() if value.ndim >= 2}
    budget -= sum(value.numel() * value.element_size() for value in params.values() if value.ndim < 2)
    costs = {name: svd_component_bytes(value) for name, value in matrices.items()}
    widths = {name: min(value.shape[0], value.numel() // value.shape[0]) for name, value in matrices.items()}
    limits = {name: min(widths[name], value.numel() * value.element_size() // costs[name])
              for name, value in matrices.items()}
    ranks = dict.fromkeys(matrices, 1)
    budget -= sum(costs.values())
    if budget < 0 or any(limit == 0 for limit in limits.values()):
        raise ValueError("Uplink budget must fit dense vectors and one SVD component per matrix")
    while True:
        # Raise the lowest retained-rank fraction that can still fit one component.
        eligible = (name for name in matrices if ranks[name] < limits[name] and costs[name] <= budget)
        name = min(eligible, key=lambda name: (ranks[name] + 1) / widths[name], default=None)
        if name is None:
            return ranks
        ranks[name] += 1
        budget -= costs[name]


class FedMuon:
    def __init__(self, params, coefficient, *, uplink_budget=None):
        self.coefficient = coefficient
        self.momentum = {name: torch.zeros_like(value) for name, value in params.items() if value.ndim >= 2}
        self.direction = {name: torch.zeros_like(value) for name, value in params.items()}
        self.momentum_ranks = None
        self.delta_ranks = None
        if uplink_budget is not None:
            self.momentum_ranks = {
                name: max(1, int(SVD_FRACTION * min(value.shape)))
                for name, value in self.momentum.items()
            }
            momentum_bytes = sum(svd_component_bytes(self.momentum[name]) * rank
                                 for name, rank in self.momentum_ranks.items())
            self.delta_ranks = model_delta_ranks(params, uplink_budget - momentum_bytes)

    def initial_state(self):
        return {name: value.clone() for name, value in self.momentum.items()}

    def downlink_bytes(self):
        return sum(v.numel() * v.element_size() for v in (*self.momentum.values(), *self.direction.values()))

    @torch.no_grad()
    def step(self, params, state, lr, weight_decay=0.0):
        for name, param in params.items():
            direction = param.grad
            if name in state:
                momentum = state[name].mul_(self.coefficient).add_(direction)
                direction = orthogonalize(momentum)
                param.mul_(1.0 - lr * weight_decay)
            param.add_(direction, alpha=-lr * (1.0 - ALIGNMENT))
            param.add_(self.direction[name], alpha=-lr * ALIGNMENT)

    def start_round(self):
        self.momentum_sum = {name: torch.zeros_like(value) for name, value in self.momentum.items()}
        self.direction_sum = {name: torch.zeros_like(value) for name, value in self.direction.items()}

    @torch.no_grad()
    def transmit_model_delta(self, payload):
        decoded = {}
        wire_bytes = 0
        for name, value in payload.items():
            rank = self.delta_ranks.get(name) if self.delta_ranks is not None else None
            if rank is not None:
                decoded[name], size = decode_truncated_svd(value, rank)
            else:
                decoded[name] = value
                size = value.numel() * value.element_size()
            wire_bytes += size
        return decoded, wire_bytes

    @torch.no_grad()
    def accept(self, payload, state, local_steps, lr):
        momentum_bytes = 0
        for name, value in state.items():
            if self.momentum_ranks is None:
                self.momentum_sum[name].add_(value)
                momentum_bytes += value.numel() * value.element_size()
                continue

            rank = self.momentum_ranks[name]
            if rank == 0:
                continue
            # Decode each client message before averaging, as in the authors' server.
            reconstructed, size = decode_truncated_svd(value, rank)
            momentum_bytes += size
            self.momentum_sum[name].add_(reconstructed)
        for name, value in payload.items():
            self.direction_sum[name].add_(value, alpha=-1.0 / (local_steps * lr))
        return momentum_bytes

    @torch.no_grad()
    def finish_round(self, participants):
        for name in self.momentum:
            self.momentum[name].copy_(self.momentum_sum[name]).div_(participants)
        for name in self.direction:
            self.direction[name].copy_(self.direction_sum[name]).div_(participants)
