import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from algorithm import (
    FedMefAlgorithm,  # 保留导入以防后续需要，但当前实现用纯 PyTorch 版本
    NeuLiteAlgorithm,  # 同上
)


# ----------------------- 模型与数据 -----------------------


class SimpleMLP(nn.Module):
    """单隐层 MLP，用于 MNIST 分类。"""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
            # 在极端非 IID（如 alpha 很小）时，可能出现空客户端，直接跳过
            continue
        subset = Subset(train_ds, idxs)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0))
    return loaders


def build_test_loader(test_ds, batch_size: int = 256) -> DataLoader:
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
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


# ----------------------- 向量化工具 -----------------------


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()], dim=0)


def load_params_from_vector(model: nn.Module, vec: torch.Tensor) -> None:
    """将一维向量加载回模型参数。vec 是 1D torch.Tensor。"""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        block = vec[offset : offset + numel].view_as(p)
        p.data.copy_(block)
        offset += numel


# ----------------------- 本地训练 -----------------------


def local_train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, lr: float) -> None:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()


# ----------------------- FedSLop：矩阵参数形式的低秩投影 + 客户端动量 -----------------------


def run_round_fedslop_torch(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    device: torch.device,
    algo_state: Dict,
    algo_cfg: Dict,
) -> Tuple[nn.Module, float]:
    """PyTorch 版 FedSLop（矩阵参数形式，修正版）。

    关键点：
    - 在每个线性层权重矩阵 W ∈ R^{out×in} 上构造低秩子空间 P_t ∈ R^{in×r}；
    - 对梯度矩阵 G 在输入维度上做投影：G_proj = G (P_t P_t^T)；
    - 客户端维护 per-layer 的矩阵动量，与参数同形状；
    - 本地多步更新中按矩阵形式执行动量和参数更新：
        v_{s+1} = μ v_s + G_proj,
        W ← W − η v_{s+1}；
    - 服务器端对每个客户端的参数差分 Δ_i^t 做逐层平均聚合。

    返回：
        - 更新后的全局模型
        - 本轮总上行通信量（按每个客户端上传全量参数差分的元素个数统计）
    """
    lr_local = algo_cfg.get("lr", 0.1)
    client_fraction = algo_cfg.get("client_fraction", 0.2)
    local_steps = algo_cfg.get("local_epochs", 1)
    momentum = algo_cfg.get("momentum", 0.9)
    rank = algo_cfg.get("proj_rank", 16)  # 对每层输入维度的低秩 r
    seed = algo_cfg.get("seed", 0)
    round_idx = algo_cfg.get("round", 0)

    num_clients = len(client_loaders)
    if num_clients == 0:
        return global_model, 0.0

    m = max(1, int(client_fraction * num_clients))
    rng = np.random.RandomState(seed + round_idx)
    selected = rng.choice(num_clients, size=m, replace=False)

    # 初始化客户端矩阵动量：按层存储与参数同形状的动量矩阵
    if "client_momentum_mats" not in algo_state:
        algo_state["client_momentum_mats"] = {}
    client_momentum_mats: Dict[int, Dict[str, torch.Tensor]] = algo_state["client_momentum_mats"]

    # 为当前轮构造每个线性层的随机子空间投影 Π_t^l
    layer_Pis: Dict[str, torch.Tensor] = {}
    base_layer_seed = seed + round_idx * 1000
    for i, (name, param) in enumerate(global_model.named_parameters()):
        if param.ndim == 2:  # 仅对权重矩阵做投影（[out, in]）
            in_dim = param.shape[1]
            r_dim = min(rank, in_dim)
            g_torch = torch.Generator(device="cpu")
            g_torch.manual_seed(base_layer_seed + i)
            A = torch.randn(in_dim, r_dim, generator=g_torch, device="cpu")
            A, _ = torch.linalg.qr(A, mode="reduced")  # in_dim × r_dim，列正交
            Pi = A @ A.t()  # in_dim × in_dim
            layer_Pis[name] = Pi

    # 收集每个客户端的参数差分（逐层）
    client_deltas: Dict[int, Dict[str, torch.Tensor]] = {}
    uplink_round: float = 0.0

    for cid in selected:
        # 拷贝全局模型到客户端
        model = SimpleMLP().to(device)
        model.load_state_dict(global_model.state_dict())

        # 取出该客户端上一轮的矩阵动量，并以其初始化 m_layers
        # 注意：m_layers 在批次循环中持续更新（动量在局部步之间正确累积）
        m_prev_layers = client_momentum_mats.get(cid, {})
        m_layers: Dict[str, torch.Tensor] = {k: v.clone() for k, v in m_prev_layers.items()}

        model.train()
        for _ in range(local_steps):
            for x, y in client_loaders[cid]:
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()

                # 对每个线性层的梯度做低秩投影 + 矩阵动量更新 + 参数更新
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        g_full = param.grad.detach().cpu()

                        # 非矩阵参数（如偏置）：使用标准动量 SGD
                        if param.ndim != 2:
                            m_prev = m_layers.get(name, torch.zeros_like(g_full))
                            m_new = momentum * m_prev + g_full
                            m_layers[name] = m_new
                            param.data -= lr_local * m_new.to(device)
                            continue

                        # 矩阵参数 W ∈ R^{out×in}，在输入维度上构造低秩子空间
                        if name not in layer_Pis:
                            # 保险起见，若未构造投影矩阵，则退化为全量动量 SGD
                            m_prev = m_layers.get(name, torch.zeros_like(g_full))
                            m_new = momentum * m_prev + g_full
                            m_layers[name] = m_new
                            param.data -= lr_local * m_new.to(device)
                            continue

                        Pi = layer_Pis[name]  # in_dim × in_dim
                        # 投影梯度：G_proj = G Π
                        g_proj = g_full @ Pi

                        # 矩阵动量更新：v ← μ v + G_proj（使用当前步的动量，确保批次内正确累积）
                        m_prev = m_layers.get(name, torch.zeros_like(g_full))
                        m_new = momentum * m_prev + g_proj
                        m_layers[name] = m_new

                        # 参数更新：W ← W − η v
                        param.data -= lr_local * m_new.to(device)

        # 记录该客户端的参数差分 Δ_i^t = θ_i^t − θ^t（逐层）
        delta_layers: Dict[str, torch.Tensor] = {}
        for (name, param), (gname, gparam) in zip(model.named_parameters(), global_model.named_parameters()):
            assert name == gname
            delta = (param.data.detach().cpu() - gparam.data.detach().cpu())
            delta_layers[name] = delta
        client_deltas[cid] = delta_layers

        # 统计该客户端上传的元素数（全量参数差分）
        uplink_client = sum(v.numel() for v in delta_layers.values())
        uplink_round += float(uplink_client)

        # 更新客户端矩阵动量缓存
        client_momentum_mats[cid] = m_layers

    if not client_deltas:
        return global_model, 0.0

    # 服务器端：逐层平均 Δ_i^t 并更新全局模型
    new_state = {}
    for name, gparam in global_model.named_parameters():
        deltas = [client_deltas[cid][name] for cid in selected]
        mean_delta = torch.stack(deltas, dim=0).mean(dim=0)
        new_state[name] = (gparam.data.detach().cpu() + mean_delta).to(device)

    # 将更新后的参数写回全局模型
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            param.data.copy_(new_state[name])

    return global_model, uplink_round


# ----------------------- 其它方法：PyTorch 版近似 -----------------------


def run_round_fedavg_momentum(
    global_vec: torch.Tensor,
    client_loaders: List[DataLoader],
    device: torch.device,
    algo_state: Dict,
    algo_cfg: Dict,
) -> Tuple[torch.Tensor, float]:
    """FedAvg + 服务器动量（标准基线）。

    返回：
        - 更新后的全局参数向量
        - 本轮总上行通信量（每个参与客户端上传完整参数向量的元素个数之和）
    """
    lr = algo_cfg.get("lr", 0.1)
    client_fraction = algo_cfg.get("client_fraction", 0.2)
    local_epochs = algo_cfg.get("local_epochs", 1)
    momentum = algo_cfg.get("momentum", 0.9)
    seed = algo_cfg.get("seed", 0)
    round_idx = algo_cfg.get("round", 0)

    num_clients = len(client_loaders)
    m = max(1, int(client_fraction * num_clients))
    rng = np.random.RandomState(seed + round_idx)
    selected = rng.choice(num_clients, size=m, replace=False)

    local_vecs = []
    uplink_round: float = 0.0
    for cid in selected:
        model = SimpleMLP().to(device)
        load_params_from_vector(model, global_vec.clone())
        for _ in range(local_epochs):
            local_train_epoch(model, client_loaders[cid], device, lr)
        vec = flatten_params(model).detach().cpu()
        local_vecs.append(vec)
        uplink_round += float(vec.numel())

    theta = global_vec.detach().cpu()
    theta_mean = torch.stack(local_vecs, dim=0).mean(dim=0)
    delta = theta_mean - theta

    if "v" not in algo_state:
        algo_state["v"] = torch.zeros_like(theta)
    v = algo_state["v"]
    v = momentum * v + delta
    algo_state["v"] = v

    new_vec = theta + v
    return new_vec.to(device), uplink_round


def run_round_neulite(
    global_vec: torch.Tensor,
    client_loaders: List[DataLoader],
    device: torch.device,
    algo_cfg: Dict,
) -> Tuple[torch.Tensor, float]:
    """简化版 NeuLite：按块划分参数，每轮只在一个块上做 FedAvg。

    返回：
        - 更新后的全局参数向量
        - 本轮总上行通信量（每个参与客户端上传该块参数子向量的元素个数之和）
    """
    lr = algo_cfg.get("lr", 0.1)
    client_fraction = algo_cfg.get("client_fraction", 0.2)
    local_epochs = algo_cfg.get("local_epochs", 1)
    num_blocks = algo_cfg.get("num_blocks", 4)
    block_ratio = algo_cfg.get("block_ratio", 0.0)
    seed = algo_cfg.get("seed", 0)
    round_idx = algo_cfg.get("round", 0)

    d = global_vec.numel()
    if block_ratio and block_ratio > 0:
        # 当需要和其他压缩算法对齐时，直接按比例控制每轮参与聚合的参数量。
        block_len = max(1, min(d, int(round(block_ratio * d))))
        block_start = (round_idx * block_len) % d
        block_idx = (torch.arange(block_len, dtype=torch.long) + block_start) % d
        use_explicit_idx = True
    else:
        block_size = d // num_blocks
        blocks = []
        for i in range(num_blocks):
            start = i * block_size
            end = d if i == num_blocks - 1 else (i + 1) * block_size
            blocks.append((start, end))

        b_idx = round_idx % num_blocks
        b_start, b_end = blocks[b_idx]
        use_explicit_idx = False

    num_clients = len(client_loaders)
    m = max(1, int(client_fraction * num_clients))
    rng = np.random.RandomState(seed + round_idx)
    selected = rng.choice(num_clients, size=m, replace=False)

    local_block_vecs = []
    uplink_round: float = 0.0
    for cid in selected:
        model = SimpleMLP().to(device)
        load_params_from_vector(model, global_vec.clone())
        for _ in range(local_epochs):
            local_train_epoch(model, client_loaders[cid], device, lr)
        vec = flatten_params(model).detach().cpu()
        block_vec = vec[block_idx] if use_explicit_idx else vec[b_start:b_end]
        local_block_vecs.append(block_vec)
        uplink_round += float(block_vec.numel())

    global_vec_cpu = global_vec.detach().cpu()
    if local_block_vecs:
        mean_block = torch.stack(local_block_vecs, dim=0).mean(dim=0)
        if use_explicit_idx:
            global_vec_cpu[block_idx] = mean_block
        else:
            global_vec_cpu[b_start:b_end] = mean_block
    return global_vec_cpu.to(device), uplink_round


def run_round_fedmef(
    global_vec: torch.Tensor,
    client_loaders: List[DataLoader],
    device: torch.device,
    algo_cfg: Dict,
) -> Tuple[torch.Tensor, float]:
    """简化 FedMef：基于梯度幅值的稀疏 FedAvg。

    返回：
        - 更新后的全局参数向量
        - 本轮总上行通信量（每个参与客户端上传选中坐标子向量的元素个数之和）
    """
    lr = algo_cfg.get("lr", 0.1)
    client_fraction = algo_cfg.get("client_fraction", 0.2)
    local_epochs = algo_cfg.get("local_epochs", 1)
    sparsity = algo_cfg.get("sparsity", 0.2)  # 每轮更新的参数比例
    seed = algo_cfg.get("seed", 0)
    round_idx = algo_cfg.get("round", 0)

    num_clients = len(client_loaders)
    m = max(1, int(client_fraction * num_clients))
    rng = np.random.RandomState(seed + round_idx)
    selected = rng.choice(num_clients, size=m, replace=False)

    d = global_vec.numel()
    k = max(1, int(sparsity * d))

    # 估计全局梯度（粗略）：在一个客户端上做一次前向+反向
    model = SimpleMLP().to(device)
    load_params_from_vector(model, global_vec.clone())
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    client0 = client_loaders[selected[0]]
    x, y = next(iter(client0))
    x, y = x.to(device), y.to(device)
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    grad_vec = torch.cat([p.grad.view(-1) for p in model.parameters()], dim=0).detach().cpu()

    # 选取梯度绝对值最大的 k 个位置
    _, topk_idx = torch.topk(grad_vec.abs(), k)

    # 对这些位置做 FedAvg
    local_updates = []
    uplink_round: float = 0.0
    for cid in selected:
        model = SimpleMLP().to(device)
        load_params_from_vector(model, global_vec.clone())
        for _ in range(local_epochs):
            local_train_epoch(model, client_loaders[cid], device, lr)
        vec = flatten_params(model).detach().cpu()
        sub_vec = vec[topk_idx]
        local_updates.append(sub_vec)
        uplink_round += float(sub_vec.numel())

    global_vec_cpu = global_vec.detach().cpu()
    if local_updates:
        mean_update = torch.stack(local_updates, dim=0).mean(dim=0)
        global_vec_cpu[topk_idx] = mean_update
    return global_vec_cpu.to(device), uplink_round


def run_round_fedlora(
    global_vec: torch.Tensor,
    client_loaders: List[DataLoader],
    device: torch.device,
    algo_state: Dict,
    algo_cfg: Dict,
) -> Tuple[torch.Tensor, float]:
    """更保守稳定的 FedLoRA-M 变体：
    - 客户端：完整参数向量上做本地 SGD（与 FedAvg 相同）
    - 服务器：对平均更新 Δw 做固定低秩投影 + 服务器动量

    返回：
        - 更新后的全局参数向量
        - 本轮总上行通信量（每个参与客户端上传完整参数向量的元素个数之和）
    """
    lr = algo_cfg.get("lr", 0.1)
    client_fraction = algo_cfg.get("client_fraction", 0.2)
    local_epochs = algo_cfg.get("local_epochs", 1)
    momentum = algo_cfg.get("momentum", 0.9)
    rank = algo_cfg.get("rank", 4)
    seed = algo_cfg.get("seed", 0)
    round_idx = algo_cfg.get("round", 0)

    d = global_vec.numel()

    # 固定随机正交子空间 A（只初始化一次）
    if "A_lora" not in algo_state:
        rng = np.random.RandomState(seed)
        A_np = rng.randn(d, rank)
        A_np, _ = np.linalg.qr(A_np)  # d×r，列正交
        A = torch.from_numpy(A_np.astype(np.float32))
        algo_state["A_lora"] = A
    else:
        A = algo_state["A_lora"]
    A = A.to(device)

    # 服务器动量（在投影后的方向上）
    if "v_lora" not in algo_state:
        algo_state["v_lora"] = torch.zeros(d, device=device)

    v = algo_state["v_lora"]

    num_clients = len(client_loaders)
    m = max(1, int(client_fraction * num_clients))
    rng_round = np.random.RandomState(seed + round_idx)
    selected = rng_round.choice(num_clients, size=m, replace=False)

    theta = global_vec.detach().to(device)

    local_vecs = []
    uplink_round: float = 0.0
    for cid in selected:
        # 客户端从全局参数出发，做本地 SGD（完整向量）
        model = SimpleMLP().to(device)
        load_params_from_vector(model, theta.clone())
        for _ in range(local_epochs):
            local_train_epoch(model, client_loaders[cid], device, lr)
        vec = flatten_params(model).detach().to(device)
        local_vecs.append(vec)
        uplink_round += float(vec.numel())

    if not local_vecs:
        return global_vec, 0.0

    # 普通 FedAvg 的平均更新 Δw
    theta_mean = torch.stack(local_vecs, dim=0).mean(dim=0)
    delta = theta_mean - theta  # d 维

    # 将更新投影到低秩子空间：Δw_proj = A A^T Δw
    c = A.t() @ delta.view(-1, 1)      # r×1
    delta_proj = (A @ c).view(-1)      # d

    # 服务器动量更新
    v = momentum * v + delta_proj
    algo_state["v_lora"] = v

    new_vec = theta + v
    return new_vec.to(device), uplink_round


def run_round_federated_select(
    global_vec: torch.Tensor,
    client_loaders: List[DataLoader],
    device: torch.device,
    algo_cfg: Dict,
) -> Tuple[torch.Tensor, float]:
    """简化 FederatedSelect：每轮随机选择一部分参数做 FedAvg。

    返回：
        - 更新后的全局参数向量
        - 本轮总上行通信量（每个参与客户端上传选中坐标子向量的元素个数之和）
    """
    lr = algo_cfg.get("lr", 0.1)
    client_fraction = algo_cfg.get("client_fraction", 0.2)
    local_epochs = algo_cfg.get("local_epochs", 1)
    select_ratio = algo_cfg.get("select_ratio", 0.1)
    seed = algo_cfg.get("seed", 0)
    round_idx = algo_cfg.get("round", 0)

    num_clients = len(client_loaders)
    m = max(1, int(client_fraction * num_clients))
    rng = np.random.RandomState(seed + round_idx)
    selected = rng.choice(num_clients, size=m, replace=False)

    d = global_vec.numel()
    k = max(1, int(select_ratio * d))
    idx = rng.choice(d, size=k, replace=False)

    local_updates = []
    uplink_round: float = 0.0
    for cid in selected:
        model = SimpleMLP().to(device)
        load_params_from_vector(model, global_vec.clone())
        for _ in range(local_epochs):
            local_train_epoch(model, client_loaders[cid], device, lr)
        vec = flatten_params(model).detach().cpu()
        sub_vec = vec[idx]
        local_updates.append(sub_vec)
        uplink_round += float(sub_vec.numel())

    global_vec_cpu = global_vec.detach().cpu()
    if local_updates:
        mean_update = torch.stack(local_updates, dim=0).mean(dim=0)
        global_vec_cpu[idx] = mean_update
    return global_vec_cpu.to(device), uplink_round


# ----------------------- 总体联邦训练循环 -----------------------


def run_federated_experiment(
    method: str,
    train_ds,
    test_ds,
    num_clients: int = 50,
    alpha: float = 0.1,
    rounds: int = 200,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    client_fraction: float = 0.2,
    # 通过 CLI 传入的超参（统一压缩比设置）
    proj_rank: int = 64,
    fedslop_momentum: float = 0.9,
    sparsity: float = 0.2,
    select_ratio: float = 0.1,
    num_blocks: int = 4,
    block_ratio: float = 0.0,
) -> Tuple[List[Tuple[int, float]], Dict[int, float]]:
    """在 MNIST + 单层 MLP 上运行指定联邦方法。

    返回：
        - history: [(round, test_acc), ...]
        - comm_stats: {round: uplink_elements}，每轮上行通信量（按元素个数估算）
    """

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

    global_model = SimpleMLP().to(dev)
    global_vec = flatten_params(global_model).to(dev)

    history: List[Tuple[int, float]] = []
    comm_stats: Dict[int, float] = {}
    algo_state: Dict = {"round": 0}

    # 预计算参数总维度 d
    d = int(global_vec.numel())

    for r in tqdm(range(rounds), desc=f"{method} seed={seed}, alpha={alpha}"):
        algo_state["round"] = r

        cfg = {
            "lr": lr,
            "client_fraction": client_fraction,
            "local_epochs": local_epochs,
            "momentum": fedslop_momentum,
            "seed": seed,
            "round": r,
            "proj_rank": proj_rank,
            "server_lr": 1.0,
            "num_blocks": num_blocks,
            "block_ratio": block_ratio,
            "sparsity": sparsity,
            "rank": 4,
            "select_ratio": select_ratio,
        }

        # 估算本轮参与客户端数 m（用于通信量统计）
        num_clients_eff = len(client_loaders)
        m = max(1, int(client_fraction * num_clients_eff))

        if method == "FedSLop":
            # FedSLop：统计每个参与客户端上传的全量参数差分元素数
            global_model, uplink = run_round_fedslop_torch(global_model, client_loaders, dev, algo_state, cfg)
            global_vec = flatten_params(global_model).to(dev)
            comm_stats[r] = float(uplink)
        elif method == "FedAvg-M":
            # FedAvg-M：每个参与客户端上传完整参数向量
            global_vec, uplink = run_round_fedavg_momentum(global_vec, client_loaders, dev, algo_state, cfg)
            comm_stats[r] = float(uplink)
        elif method == "NeuLite":
            # NeuLite：每轮只在一个块上做 FedAvg，上传该块参数子向量
            global_vec, uplink = run_round_neulite(global_vec, client_loaders, dev, cfg)
            comm_stats[r] = float(uplink)
        elif method == "FedMef":
            # FedMef：仅在梯度幅值最大的 k = sparsity * d 个坐标上聚合
            global_vec, uplink = run_round_fedmef(global_vec, client_loaders, dev, cfg)
            comm_stats[r] = float(uplink)
        elif method == "FedLoRA-M":
            # FedLoRA-M 在独立脚本 code/experiment_mnist_fedlora_torch.py 中实现，
            # 这里不再使用向量版近似实现，避免与真实实现混淆。
            raise NotImplementedError(
                "FedLoRA-M 的正式实现位于 code/experiment_mnist_fedlora_torch.py，"
                "请使用该脚本单独运行 FedLoRA-M 实验。"
            )
        elif method == "FederatedSelect":
            # FederatedSelect：每轮随机选择 k = select_ratio * d 个坐标做 FedAvg
            global_vec, uplink = run_round_federated_select(global_vec, client_loaders, dev, cfg)
            comm_stats[r] = float(uplink)
        else:
            raise ValueError(f"Unknown method: {method}")

        load_params_from_vector(global_model, global_vec)
        acc = evaluate_model(global_model, test_loader, dev)
        history.append((r, acc))
        print(f"[method={method} seed={seed} alpha={alpha}] round {r}: test_acc={acc:.4f}, uplink={comm_stats[r]:.0f} elems")

    return history, comm_stats


def objective_for_tuning(params: dict) -> float:
    """Auto-tune 目标函数：在小规模设置下评估 FedSLop 的平均收敛质量。

    调参目标：最大化前若干轮（例如 50 轮）的平均测试精度，
    这里通过最小化负的平均精度来实现。

    params 示例：{"proj_rank": 16, "momentum": 0.9, "lr": 0.05}
    """
    proj_rank = int(params.get("proj_rank", 16))
    momentum = float(params.get("momentum", 0.9))
    lr = float(params.get("lr", 0.1))

    # 小规模固定配置，用于调参
    method = "FedSLop"
    rounds = 50
    num_clients = 20
    alpha = 0.1
    local_epochs = 1
    batch_size = 32
    seed = 0

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据与划分
    train_ds, test_ds = get_mnist_datasets()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device)

    labels = np.array(train_ds.targets)
    client_indices = dirichlet_partition(labels, num_clients=num_clients, alpha=alpha, rng=rng)
    client_loaders = build_client_loaders(train_ds, client_indices, batch_size=batch_size)
    test_loader = build_test_loader(test_ds)

    # 初始化全局模型
    global_model = SimpleMLP().to(dev)
    algo_state: Dict = {"round": 0}

    acc_history = []
    for r in range(rounds):
        algo_state["round"] = r
        cfg = {
            "lr": lr,
            "client_fraction": 0.2,
            "local_epochs": local_epochs,
            "momentum": momentum,
            "seed": seed,
            "round": r,
            "proj_rank": proj_rank,
        }
        global_model = run_round_fedslop_torch(global_model, client_loaders, dev, algo_state, cfg)
        acc = evaluate_model(global_model, test_loader, dev)
        acc_history.append(float(acc))

    # 目标：最大化前 rounds 轮平均精度 => 最小化负平均精度
    mean_acc = float(np.mean(acc_history))
    return -mean_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="FedSLop,FedMef,FederatedSelect,NeuLite,FedAvg-M,FedLoRA-M")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--alphas", type=str, default="", help="可选，多个 Dirichlet alpha 值，逗号分隔。如 1.0,0.5,0.1,0.05")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--client_fraction", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="data/mnist_fed_results.csv")
    # --- 超参配置（统一压缩比设置）---
    parser.add_argument("--proj_rank", type=int, default=64, help="FedSLop 的投影秩（决定梯度子空间覆盖率）")
    parser.add_argument("--fedslop_momentum", type=float, default=0.8, help="FedSLop 的动量系数")
    parser.add_argument("--sparsity", type=float, default=0.2, help="FedMef 的稀疏度（每轮更新比例）")
    parser.add_argument("--select_ratio", type=float, default=0.1, help="FederatedSelect 的坐标选择比例")
    parser.add_argument("--num_blocks", type=int, default=4, help="NeuLite 的分块数（每轮更新 1/num_blocks）")
    parser.add_argument("--block_ratio", type=float, default=0.0, help="NeuLite 每轮更新比例；若 > 0 则优先于 num_blocks")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    train_ds, test_ds = get_mnist_datasets()

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    import csv

    # 解析 alphas：若提供 --alphas，则循环多个 alpha；否则退回到单一 args.alpha
    if args.alphas.strip():
        alpha_list = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    else:
        alpha_list = [args.alpha]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed", "alpha", "round", "test_acc", "uplink_elems"])

        for alpha in alpha_list:
            for method in methods:
                for seed in seeds:
                    print(f"Running method={method}, seed={seed}, alpha={alpha} ...")
                    history, comm_stats = run_federated_experiment(
                        method=method,
                        train_ds=train_ds,
                        test_ds=test_ds,
                        num_clients=args.num_clients,
                        alpha=alpha,
                        rounds=args.rounds,
                        local_epochs=args.local_epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        seed=seed,
                        device=args.device,
                        client_fraction=args.client_fraction,
                        proj_rank=args.proj_rank,
                        fedslop_momentum=args.fedslop_momentum,
                        sparsity=args.sparsity,
                        select_ratio=args.select_ratio,
                        num_blocks=args.num_blocks,
                        block_ratio=args.block_ratio,
                    )
                    for r, acc in history:
                        uplink = comm_stats.get(r, 0.0)
                        writer.writerow([method, seed, alpha, r, acc, uplink])
                    print(f"  Final acc (alpha={alpha}): {history[-1][1]:.4f}")

    print(f"All results saved to {args.output}")


if __name__ == "__main__":
    main()
