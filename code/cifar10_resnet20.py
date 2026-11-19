"""Matched, full-participation CIFAR-10 comparison for the FedSLoP revision."""

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torchvision.datasets import CIFAR10


class ResidualBlock(nn.Module):
    def __init__(self, source_width, width, stride):
        super().__init__()
        self.stride = stride
        self.width = width
        self.conv1 = nn.Conv2d(source_width, width, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

    def forward(self, x):
        shortcut = x
        if self.stride != 1:
            shortcut = shortcut[:, :, ::2, ::2]
            channel_pad = (self.width - shortcut.size(1)) // 2
            shortcut = F.pad(shortcut, (0, 0, 0, 0, channel_pad, channel_pad))
        hidden = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(hidden)) + shortcut, inplace=True)


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        blocks = []
        source_width = 16
        for width in (16, 32, 64):
            for _ in range(3):
                blocks.append(ResidualBlock(source_width, width, 1 if source_width == width else 2))
                source_width = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(64, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.blocks(self.stem(x)).mean(dim=(2, 3)))


class LoRAWeight(nn.Module):
    """A frozen convolutional base plus two trainable low-rank factors."""

    def __init__(self, weight, rank):
        super().__init__()
        width = weight.shape[0]
        input_size = weight.numel() // width
        self.A = nn.Parameter(weight.new_empty((width, rank)))
        self.B = nn.Parameter(weight.new_zeros((rank, input_size)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        return weight + (self.A @ self.B).reshape_as(weight)


def add_conv_lora(model, rank_ratio):
    ranks = {}
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, nn.Conv2d):
            continue
        width = layer.weight.shape[0]
        input_size = layer.weight.numel() // width
        target_payload = max(1, int(width * rank_ratio)) * input_size
        rank = max(1, target_payload // (width + input_size))
        layer.weight.requires_grad_(False)
        parametrize.register_parametrization(layer, "weight", LoRAWeight(layer.weight, rank))
        ranks[name] = rank
    return ranks


def tensor_bytes(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def seeded_generator(*parts):
    seed = int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0])
    return torch.Generator(device="cpu").manual_seed(seed)


def split_clients(labels, seed, num_clients, alpha):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    train_by_class, val_by_class = [], []
    for label in range(10):
        class_indices = rng.permutation(np.flatnonzero(labels == label))
        val_by_class.append(class_indices[:500])
        train_by_class.append(class_indices[500:])

    for _ in range(100):
        clients = [[] for _ in range(num_clients)]
        for indices in train_by_class:
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(num_clients, alpha)))
            for shard, indices_for_client in zip(clients, np.split(indices, np.cumsum(counts)[:-1])):
                shard.extend(indices_for_client.tolist())
        if min(map(len, clients)) >= 64:
            return clients, np.concatenate(val_by_class)
    raise RuntimeError("Unable to obtain 64 samples per client at the requested concentration")


class Benchmark:
    def __init__(self, data_root, args, device):
        train = CIFAR10(data_root, train=True, download=False)
        test = CIFAR10(data_root, train=False, download=False)
        self.device = device
        self.args = args
        self.clients, val_indices = split_clients(np.asarray(train.targets), args.seed, args.clients, args.alpha)
        self.train_images = torch.from_numpy(train.data).to(device)
        self.train_labels = torch.tensor(train.targets, device=device)
        self.val_set = self.prepare_eval(train.data[val_indices], np.asarray(train.targets)[val_indices])
        self.test_set = self.prepare_eval(test.data, test.targets)
        self.client_ids = [torch.tensor(ids, device=device) for ids in self.clients]
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616], device=device)
        self.pixels = torch.arange(32, device=device)[None, :]
        self.batch_axis = torch.arange(args.batch_size, device=device)[:, None, None]

    def prepare_eval(self, data, targets):
        images = torch.from_numpy(data).to(self.device)
        labels = torch.as_tensor(np.asarray(targets), device=self.device)
        return images, labels

    def normalize(self, images):
        images = (images.float() / 255.0 - self.mean) / self.std
        return images.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)

    def batches(self, client_id, round_id):
        args = self.args
        generator = seeded_generator(args.seed, 1, round_id, client_id)
        client_ids = self.client_ids[client_id]
        indices = torch.randint(len(client_ids), (args.local_steps, args.batch_size), generator=generator)
        offsets = torch.randint(9, (args.local_steps, 2, args.batch_size), generator=generator).to(self.device)
        flipped = torch.randint(2, (args.local_steps, args.batch_size, 1), generator=generator).to(self.device)
        for step in range(args.local_steps):
            ids = client_ids[indices[step].to(self.device)]
            images = self.train_images[ids]
            padded = F.pad(images, (0, 0, 4, 4, 4, 4))
            rows = (self.pixels + offsets[step, 0, :, None])[:, :, None]
            columns = torch.where(flipped[step].bool(), 31 - self.pixels, self.pixels)
            columns = (columns + offsets[step, 1, :, None])[:, None, :]
            images = padded[self.batch_axis, rows, columns]
            yield self.normalize(images), self.train_labels[ids]

    @torch.no_grad()
    def evaluate(self, model, test=False):
        model.eval()
        images, labels = self.test_set if test else self.val_set
        correct = torch.zeros((), device=self.device)
        loss = torch.zeros((), device=self.device)
        for batch, target in zip(images.split(256), labels.split(256)):
            logits = model(self.normalize(batch))
            loss += F.cross_entropy(logits, target, reduction="sum")
            correct += logits.argmax(dim=1).eq(target).sum()
        return correct.item() / len(labels), loss.item() / len(labels)


class FederatedComparison:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        torch.manual_seed(args.seed)
        self.model = ResNet20().to(device=device, memory_format=torch.channels_last)
        self.lora_ranks = {}
        if args.method == "FedLoRA-M":
            self.lora_ranks = add_conv_lora(self.model, args.rank_ratio)
            self.clientside = deepcopy(self.model)
        else:
            self.clientside = ResNet20().to(device=device, memory_format=torch.channels_last)
        self.params = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        self.local_params = {name: p for name, p in self.clientside.named_parameters() if p.requires_grad}
        self.frozen_base_bytes = tensor_bytes(p for p in self.model.parameters() if not p.requires_grad)
        self.buffers = dict(self.model.named_buffers())
        self.local_buffers = dict(self.clientside.named_buffers())
        self.ranks = {}
        for name, param in self.params.items():
            if param.ndim != 4 or args.method in ("FedAvg-M", "FedLoRA-M", "FedMuon", "FedMuon-SVD"):
                continue
            width = param.shape[0]
            rank = max(1, int(width * args.rank_ratio))
            if args.method != "FedSLoP" or rank < width:
                self.ranks[name] = rank
        self.use_server_momentum = args.method in ("FedAvg-M", "FedLoRA-M") or args.fedslop_server_momentum
        self.server_momentum = {}
        self.client_dual = [{} for _ in range(args.clients)]
        self.bases = self.draw_bases(0)
        self.muon = None
        if args.method in ("FedMuon", "FedMuon-SVD"):
            from fedmuon_matched import FedMuon
            self.muon = FedMuon(
                self.params, args.client_momentum, compress_momentum=args.method == "FedMuon-SVD",
                svd_rank_rule=getattr(args, "svd_rank_rule", "payload"),
                svd_fraction=getattr(args, "svd_fraction", 0.05),
                uplink_budget=getattr(args, "svd_uplink_budget", None),
                buffer_bytes=tensor_bytes(self.local_buffers.values()),
            )

    def draw_bases(self, round_id):
        generator = seeded_generator(self.args.seed, 2, round_id)
        bases = {}
        for name, rank in self.ranks.items():
            width = self.params[name].shape[0]
            if self.args.method == "FedSLoP":
                g = torch.randn((width, rank), generator=generator)
                q, r = torch.linalg.qr(g, mode="reduced")
                signs = torch.where(r.diag() < 0, -1.0, 1.0)
                basis = q * signs
            else:
                ids = torch.randperm(width, generator=generator)[:rank]
                signs = (torch.randint(2, (rank,), generator=generator) * 2 - 1).float()
                basis = torch.zeros(width, rank)
                basis[ids, torch.arange(rank)] = math.sqrt(width / rank) * signs
            bases[name] = basis.to(self.device)
        return bases

    @torch.no_grad()
    def load_global_model(self):
        for name, param in self.local_params.items():
            param.copy_(self.params[name])
        for name, buffer in self.local_buffers.items():
            buffer.copy_(self.buffers[name])
        self.clientside.train()

    @torch.no_grad()
    def local_step(self, increments, momentum, dual, lr):
        if self.muon:
            self.muon.step(self.local_params, momentum, lr, self.args.weight_decay)
            return
        for name, param in self.local_params.items():
            gradient = param.grad
            if param.ndim == 4 or name == "classifier.weight" or (self.lora_ranks and param.ndim == 2):
                gradient = gradient.add(param, alpha=self.args.weight_decay)
            if name in self.bases:
                basis = self.bases[name]
                coords = basis.T @ gradient.reshape(param.shape[0], -1)
                if self.args.method == "FedSLoP":
                    momentum[name].mul_(self.args.client_momentum).add_(coords)
                    if self.args.inplace_projected_step:
                        param.add_((basis @ momentum[name]).reshape_as(param), alpha=-lr)
                        continue
                    increments[name].add_(momentum[name], alpha=-lr)
                else:
                    increments[name].add_(coords, alpha=-lr * self.ranks[name] / param.shape[0])
                    if name in dual:
                        increments[name].add_(dual[name], alpha=-1.0 / self.args.local_steps)
                delta = (basis @ increments[name]).reshape_as(param)
                param.copy_(self.params[name]).add_(delta)
            elif self.args.method == "FedSLoP":
                momentum[name].mul_(self.args.client_momentum).add_(gradient)
                param.add_(momentum[name], alpha=-lr)
            else:
                param.add_(gradient, alpha=-lr)

    def make_client_state(self):
        increments = {}
        if not self.args.inplace_projected_step:
            increments = {
                name: param.new_zeros((rank, param.numel() // param.shape[0]))
                for name, rank in self.ranks.items()
                for param in (self.params[name],)
            }
        momentum = {}
        if self.args.method == "FedSLoP":
            momentum = {
                name: param.new_zeros((self.ranks[name], param.numel() // param.shape[0]))
                if name in self.ranks else torch.zeros_like(param)
                for name, param in self.local_params.items()
            }
        if self.muon:
            momentum = self.muon.initial_state()
        return increments, momentum

    @torch.no_grad()
    def encode_client_upload(self, increments):
        upload = {}
        for name, param in self.local_params.items():
            if name in increments:
                upload[name] = increments[name]
            else:
                delta = param - self.params[name]
                if name in self.bases:
                    delta = self.bases[name].T @ delta.reshape(param.shape[0], -1)
                upload[name] = delta
        return upload

    def train_round(self, benchmark, round_id, lr):
        args = self.args
        parameter_sum = {}
        buffer_sum = {}
        selected_updates = []
        total_uplink = 0
        loss_sum = torch.zeros((), device=self.device)
        downlink = tensor_bytes(self.params.values()) + tensor_bytes(self.buffers.values())
        if self.bases:
            downlink += 8
        if round_id == 0:
            downlink += self.frozen_base_bytes
        if self.muon:
            downlink += self.muon.downlink_bytes()
            self.muon.start_round()

        for client_id in range(args.clients):
            self.load_global_model()
            increments, momentum = self.make_client_state()
            for images, labels in benchmark.batches(client_id, round_id):
                self.clientside.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.clientside(images), labels)
                loss.backward()
                loss_sum += loss.detach()
                self.local_step(increments, momentum, self.client_dual[client_id], lr)

            with torch.no_grad():
                upload = self.encode_client_upload(increments)
                total_uplink += tensor_bytes(self.local_buffers.values())
                if self.muon:
                    upload, sent_bytes = self.muon.transmit_model_delta(upload)
                    total_uplink += sent_bytes
                    total_uplink += self.muon.accept(upload, momentum, args.local_steps, lr)
                else:
                    total_uplink += tensor_bytes(upload.values())
                if args.method == "FedSub-CD":
                    selected_updates.append(increments)
                for name, value in upload.items():
                    if name not in parameter_sum:
                        parameter_sum[name] = value.clone()
                    else:
                        parameter_sum[name].add_(value)
                for name, value in self.local_buffers.items():
                    if name not in buffer_sum:
                        buffer_sum[name] = value.clone()
                    elif value.is_floating_point():
                        buffer_sum[name].add_(value)

        with torch.no_grad():
            for name, value in parameter_sum.items():
                value.div_(args.clients)
                if name in self.bases:
                    full_increment = (self.bases[name] @ value).reshape_as(self.params[name])
                else:
                    full_increment = value
                if self.use_server_momentum:
                    # Keep cross-round momentum in full coordinates because the basis changes.
                    if name not in self.server_momentum:
                        self.server_momentum[name] = torch.zeros_like(full_increment)
                    full_increment = self.server_momentum[name].mul_(args.server_momentum).add_(full_increment)
                self.params[name].add_(full_increment, alpha=args.server_stepsize)
            for name, buf in self.buffers.items():
                value = buffer_sum[name]
                if value.is_floating_point():
                    value.div_(args.clients)
                buf.copy_(value)

            if self.muon:
                self.muon.finish_round(args.clients)
            if self.bases:
                next_bases = self.draw_bases(round_id + 1)
                if args.method == "FedSub-CD":
                    downlink += tensor_bytes(parameter_sum[name] for name in self.bases)
                    for name, basis in self.bases.items():
                        rotate = next_bases[name].T @ basis
                        for client_id, coords in enumerate(selected_updates):
                            old_dual = self.client_dual[client_id].get(name)
                            adjusted = coords[name] - parameter_sum[name]
                            if old_dual is not None:
                                adjusted.add_(old_dual)
                            self.client_dual[client_id][name] = rotate @ adjusted
                self.bases = next_bases

        train_loss = loss_sum.item() / (args.clients * args.local_steps)
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Nonfinite training loss at round {round_id + 1}")
        return train_loss, total_uplink, downlink * args.clients


def run(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    benchmark = Benchmark(args.data_root, args, device)
    training = FederatedComparison(args, device)
    args.output.mkdir(parents=True, exist_ok=False)

    metadata = vars(args).copy()
    metadata["parameters"] = sum(param.numel() for param in training.params.values())
    metadata["ranks"] = training.ranks
    if args.method == "FedLoRA-M":
        metadata["lora_ranks"] = training.lora_ranks
        metadata["frozen_base_bytes"] = training.frozen_base_bytes
    if args.method == "FedSLoP":
        metadata["client_momentum_bytes"] = sum(
            param.element_size() * param.numel() * training.ranks.get(name, param.shape[0]) // param.shape[0]
            for name, param in training.params.items()
        )
        metadata["client_basis_bytes"] = tensor_bytes(training.bases.values())
    if training.muon:
        metadata["fedmuon_momentum_svd_ranks"] = training.muon.momentum_ranks
        metadata["fedmuon_delta_svd_ranks"] = training.muon.delta_ranks
    metadata["client_sizes"] = list(map(len, benchmark.clients))
    metadata["torch_version"] = torch.__version__
    metadata["cuda_device"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    columns = ["round", "lr", "train_loss", "evaluation_split", "accuracy", "eval_loss",
               "uplink_bytes", "downlink_bytes", "cumulative_uplink_bytes", "cumulative_downlink_bytes",
               "elapsed_seconds"]
    total_uplink = total_downlink = 0
    start = time.monotonic()
    with (args.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for t in range(args.rounds):
            lr = args.lr
            if not args.tuning:
                lr *= 0.1 ** int(t >= int(args.rounds * 0.6))
                lr *= 0.1 ** int(t >= int(args.rounds * 0.85))
            train_loss, uplink, downlink = training.train_round(benchmark, t, lr)
            total_uplink += uplink
            total_downlink += downlink
            evaluate = (t + 1) % args.eval_every == 0 or t + 1 == args.rounds
            accuracy, eval_loss = ("", "")
            split = ""
            if evaluate:
                validation = args.tuning or args.validation
                accuracy, eval_loss = benchmark.evaluate(training.model, test=not validation)
                if not math.isfinite(eval_loss):
                    raise FloatingPointError(f"Nonfinite evaluation loss at round {t + 1}")
                split = "validation" if validation else "test"
                print(f"{args.method} seed={args.seed} lr={lr:.5f} round={t+1}/{args.rounds} "
                      f"{split}_acc={accuracy:.4f} uplink={total_uplink:,} "
                      f"seconds={time.monotonic()-start:.1f}", flush=True)
            writer.writerow([t + 1, lr, train_loss, split, accuracy, eval_loss, uplink, downlink, total_uplink,
                             total_downlink, round(time.monotonic() - start, 3)])
            file.flush()
    (args.output / "done").touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", choices=("FedSLoP", "FedAvg-M", "FedSub-CD", "FedLoRA-M", "FedMuon", "FedMuon-SVD"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--rank-ratio", type=float, default=0.125)
    parser.add_argument("--svd-rank-rule", choices=("payload", "matrix"), default="payload")
    parser.add_argument("--svd-fraction", type=float, default=0.05)
    parser.add_argument("--svd-uplink-budget", type=int, help="Maximum bytes per client and round for both SVD messages.")
    parser.add_argument("--client-momentum", type=float, default=0.8)
    parser.add_argument("--server-momentum", type=float, default=0.9)
    parser.add_argument("--fedslop-server-momentum", action="store_true",
                        help="Apply the configured server momentum to reconstructed FedSLoP updates.")
    parser.add_argument("--server-stepsize", type=float, default=1.0,
                        help="Server multiplier for the averaged FedSLoP update.")
    parser.add_argument("--inplace-projected-step", action="store_true",
                        help="Update full weights from projected momentum and encode the delta at upload.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument("--tuning", action="store_true", help="Use validation and a constant learning rate.")
    evaluation.add_argument("--validation", action="store_true", help="Use validation with the regular LR schedule.")
    args = parser.parse_args()
    if not 0 < args.svd_fraction <= 1:
        parser.error("--svd-fraction must be in (0, 1]")
    if args.svd_uplink_budget is not None and (args.svd_uplink_budget <= 0 or args.method != "FedMuon-SVD"):
        parser.error("--svd-uplink-budget requires a positive byte budget and --method FedMuon-SVD")
    if not (0 < args.rank_ratio <= 1):
        parser.error("--rank-ratio must be in (0, 1]")
    if not (0 <= args.client_momentum < 1):
        parser.error("--client-momentum must be in [0, 1)")
    if not (0 <= args.server_momentum < 1):
        parser.error("--server-momentum must be in [0, 1)")
    if not (math.isfinite(args.server_stepsize) and args.server_stepsize > 0):
        parser.error("--server-stepsize must be positive and finite")
    if args.server_stepsize != 1.0 and args.method != "FedSLoP":
        parser.error("A non-unit --server-stepsize requires FedSLoP")
    if args.fedslop_server_momentum and args.method != "FedSLoP":
        parser.error("--fedslop-server-momentum requires FedSLoP")
    if args.inplace_projected_step and args.method != "FedSLoP":
        parser.error("--inplace-projected-step requires FedSLoP")
    run(args)


if __name__ == "__main__":
    main()
