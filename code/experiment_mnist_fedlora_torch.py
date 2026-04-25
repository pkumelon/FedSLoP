import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


# ----------------------- LoRA MLP 模型 -----------------------


class LoRAMLP(nn.Module):
    """带 LoRA 结构的单隐层 MLP，用于 MNIST 分类。

    - 对 fc1, fc2 的权重引入低秩增量 A @ B
    - 底座权重 fc1.weight, fc2.weight 作为共享基座，客户端仅更新 LoRA 参数
    """

    def __init__(self, hidden_dim: int = 128, rank: int = 4):
        super().__init__()
        # 底座权重
        self.fc1 = nn.Linear(28 * 28, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 10, bias=True)

        self.rank = rank
        in1, out1 = self.fc1.in_features, self.fc1.out_features
        in2, out2 = self.fc2.in_features, self.fc2.out_features

        # LoRA for fc1: A1 (out1 x rank), B1 (rank x in1)
        self.A1 = nn.Parameter(torch.zeros(out1, rank))
        self.B1 = nn.Parameter(torch.zeros(rank, in1))

        # LoRA for fc2: A2 (out2 x rank), B2 (rank x in2)
        self.A2 = nn.Parameter(torch.zeros(out2, rank))
        self.B2 = nn.Parameter(torch.zeros(rank, in2))

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # LoRA 一般小初始化，这里给 A 用 Kaiming，B 用 0
        nn.init.kaiming_uniform_(self.A1, a=np.sqrt(5))
        nn.init.zeros_(self.B1)
        nn.init.kaiming_uniform_(self.A2, a=np.sqrt(5))
        nn.init.zeros_(self.B2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        # fc1: (W1 + A1 B1) x + b1
        W1_eff = self.fc1.weight + self.A1 @ self.B1
        x = F.linear(x, W1_eff, self.fc1.bias)
        x = F.relu(x)

        # fc2: (W2 + A2 B2) x + b2
        W2_eff = self.fc2.weight + self.A2 @ self.B2
        x = F.linear(x, W2_eff, self.fc2.bias)
        return x


# ----------------------- 数据与划分 -----------------------


def get_mnist_datasets(data_dir: str = "data/torch_datasets"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_ds, test_ds


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, rng: np.random.RandomState):
    """按标签做 Dirichlet 非 IID 划分。返回每个客户端的样本索引列表。"""
    num_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = idx_by_class[c]
        rng.shuffle(idx_c)
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    return client_indices


def build_client_loaders(train_ds, client_indices, batch_size: int) -> List[DataLoader]:
    loaders: List[DataLoader] = []
    for idxs in client_indices:
        if len(idxs) == 0:
            # 极端非 IID（alpha 很小）时可能出现空客户端，跳过
            continue
        subset = Subset(train_ds, idxs)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0))
    return loaders


def build_test_loader(test_ds, batch_size: int = 256) -> DataLoader:
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ----------------------- LoRA 参数工具 -----------------------


def get_lora_params(model: LoRAMLP) -> Dict[str, torch.Tensor]:
    return {
        "A1": model.A1.detach().cpu().clone(),
        "B1": model.B1.detach().cpu().clone(),
        "A2": model.A2.detach().cpu().clone(),
        "B2": model.B2.detach().cpu().clone(),
    }


def set_lora_params(model: LoRAMLP, lora_state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        model.A1.copy_(lora_state["A1"])
        model.B1.copy_(lora_state["B1"])
        model.A2.copy_(lora_state["A2"])
        model.B2.copy_(lora_state["B2"])


# ----------------------- 本地训练与评估 -----------------------


def local_train_lora(
    model: LoRAMLP,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_epochs: int,
) -> None:
    """客户端本地训练：只更新 LoRA 参数，底座权重冻结。"""
    model.to(device)
    model.train()

    # 冻结底座权重，只训练 LoRA 参数
    for p in model.fc1.parameters():
        p.requires_grad = False
    for p in model.fc2.parameters():
        p.requires_grad = False

    # 只优化 LoRA 参数
    optimizer = torch.optim.SGD(
        [model.A1, model.B1, model.A2, model.B2],
        lr=lr,
    )

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_model(model: LoRAMLP, test_loader: DataLoader, device: torch.device) -> float:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


# ----------------------- FedLoRA-M 主循环 -----------------------


def run_fedlora_experiment(
    train_ds,
    test_ds,
    num_clients: int = 50,
    alpha: float = 0.1,
    rounds: int = 100,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
    rank: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> List[Tuple[int, float]]:
    """在 MNIST 上运行 FedLoRA-M，返回 (round, test_acc) 列表。"""

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.manual_seed_all(seed)
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    labels = np.array(train_ds.targets)
    client_indices = dirichlet_partition(labels, num_clients=num_clients, alpha=alpha, rng=rng)
    client_loaders = build_client_loaders(train_ds, client_indices, batch_size=batch_size)
    test_loader = build_test_loader(test_ds)

    # 记录有效客户端数量（过滤掉空客户端后）
    num_effective_clients = len(client_loaders)
    if num_effective_clients == 0:
        raise RuntimeError("No non-empty clients available after Dirichlet partition. Check alpha setting.")

    # 初始化全局模型（底座 + LoRA）
    global_model = LoRAMLP(hidden_dim=128, rank=rank).to(dev)

    # 服务器端动量（针对 LoRA 参数）
    momentum = 0.9
    global_lora = get_lora_params(global_model)
    v_lora = {k: torch.zeros_like(v) for k, v in global_lora.items()}

    history: List[Tuple[int, float]] = []

    client_fraction = 0.2

    for r in tqdm(range(rounds), desc=f"FedLoRA-M seed={seed}, alpha={alpha}"):
        # 每轮根据有效客户端数量重新采样参与客户端
        m = max(1, int(client_fraction * num_effective_clients))
        selected = rng.choice(num_effective_clients, size=m, replace=False)

        local_loras = []

        for cid in selected:
            # 拷贝全局模型
            model = LoRAMLP(hidden_dim=128, rank=rank).to(dev)
            model.load_state_dict(global_model.state_dict())
            set_lora_params(model, global_lora)

            # 本地训练：只更新 LoRA
            local_train_lora(model, client_loaders[cid], dev, lr, local_epochs)

            # 提取本地 LoRA 参数
            local_lora = get_lora_params(model)
            local_loras.append(local_lora)

        # 服务器端：FedAvg + 动量聚合 LoRA 参数
        if local_loras:
            mean_lora: Dict[str, torch.Tensor] = {}
            for k in global_lora.keys():
                stacked = torch.stack([l[k] for l in local_loras], dim=0)
                mean_lora[k] = stacked.mean(dim=0)

            # 动量更新：v = μ v + (mean - global)
            for k in global_lora.keys():
                delta = mean_lora[k] - global_lora[k]
                v_lora[k] = momentum * v_lora[k] + delta
                global_lora[k] = global_lora[k] + v_lora[k]

            # 将更新后的 LoRA 参数写回全局模型
            set_lora_params(global_model, global_lora)

        # 评估
        acc = evaluate_model(global_model, test_loader, dev)
        history.append((r, acc))
        print(f"[FedLoRA-M seed={seed}] round {r}: test_acc={acc:.4f}")

    return history


# ----------------------- 主函数 -----------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--alphas", type=str, default="", help="可选，多个 Dirichlet alpha 值，逗号分隔。如 1.0,0.5,0.1,0.05")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="data/mnist_fed_results_fedlora_m.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    train_ds, test_ds = get_mnist_datasets()

    seeds = [int(s) for s in args.seeds.split(",")]

    import csv

    # 解析 alphas：若提供 --alphas，则循环多个 alpha；否则使用单一 args.alpha
    if args.alphas.strip():
        alpha_list = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    else:
        alpha_list = [args.alpha]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed", "alpha", "round", "test_acc"])

        for alpha in alpha_list:
            for seed in seeds:
                print(f"Running FedLoRA-M, seed={seed}, alpha={alpha} ...")
                history = run_fedlora_experiment(
                    train_ds=train_ds,
                    test_ds=test_ds,
                    num_clients=args.num_clients,
                    alpha=alpha,
                    rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    rank=args.rank,
                    seed=seed,
                    device=args.device,
                )
                for r, acc in history:
                    writer.writerow(["FedLoRA-M", seed, alpha, r, acc])
                print(f"  Final acc (alpha={alpha}): {history[-1][1]:.4f}")

    print(f"FedLoRA-M results saved to {args.output}")


if __name__ == "__main__":
    main()
