# FedSLop

Paper: *FedSLoP: Memory-Efficient Federated Learning with Low-Rank Gradient Projection*.

This repository provides minimal code to reproduce the experiments reported in the paper.

## Introduction

Federated learning (FL) trains models across many clients without sharing raw data, but practical deployments must cope with data heterogeneity and strict resource constraints on edge devices. A recurring bottleneck is the client-side optimizer state (e.g., momentum buffers), which is full-dimensional for modern models and can dominate memory usage.

FedSLoP is a FedAvg-style federated optimization method that integrates **random low-rank subspace projections** with **momentum** to improve memory efficiency while staying compatible with the standard FL workflow. At each communication round, the algorithm samples a fresh low-dimensional (Stiefel) subspace and constrains local stochastic gradients and momentum updates to this subspace. This reduces the effective dimension of optimizer state and update directions **without modifying the underlying model architecture**.

**Main takeaways:**

- **Theory**: Under standard smoothness and bounded-variance assumptions, FedSLoP is guaranteed to converge to a first-order stationary point, with a nonconvex rate $O(1/\sqrt{NT})$.
- **Accuracy under compression**: With a reported compression ratio of $1/7$, FedSLoP reaches 0.9511±0.0028 final test accuracy at $\alpha=0.1$ (T=100), close to FedAvg-M at 0.9551±0.0015, and higher than sparse baselines and FedLoRA-M (0.9247±0.0063).
- **Robustness to heterogeneity**: At $\alpha=0.05$, FedSLoP achieves 0.9269 final accuracy versus 0.9035 for FedAvg-M (≈2% absolute improvement).

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

If `torch/torchvision` installation fails on your machine (OS/CPU/GPU mismatch), follow the official PyTorch install guide to select a compatible build.

MNIST is downloaded automatically via `torchvision.datasets.MNIST(..., download=True)` into `data/torch_datasets/`.

## Reproducing experiments

### 1) Main comparison on non-IID MNIST ($\alpha=0.1$)

Run the baselines and FedSLoP (excluding FedLoRA-M):

```powershell
python code\experiment_mnist_federated_torch.py `
  --methods FedSLop,FedMef,FederatedSelect,NeuLite,FedAvg-M `
  --rounds 100 --num_clients 50 --alpha 0.1 `
  --local_epochs 1 --batch_size 32 --lr 0.018 `
  --client_fraction 0.2 --seeds 0,1,2 --device cpu `
  --proj_rank 112 --fedslop_momentum 0.8 `
  --sparsity 0.15 --select_ratio 0.15 --block_ratio 0.15 `
  --output data\mnist_main_methods.csv
```

Run FedLoRA-M separately:

```powershell
python code\experiment_mnist_fedlora_torch.py `
  --rounds 100 --num_clients 50 --alpha 0.1 `
  --local_epochs 1 --batch_size 32 --lr 0.018 `
  --rank 15 --seeds 0,1,2 --device cpu `
  --output data\mnist_main_fedlora.csv
```

### 2) Robustness to heterogeneity (Dirichlet $\alpha$ sweep)

Baselines and FedSLoP:

```powershell
python code\experiment_mnist_federated_torch.py `
  --methods FedSLop,FedMef,FederatedSelect,NeuLite,FedAvg-M `
  --rounds 100 --num_clients 50 --alphas 0.05,0.1,0.5,1.0 `
  --local_epochs 1 --batch_size 32 --lr 0.018 `
  --client_fraction 0.2 --seeds 0,1,2 --device cpu `
  --proj_rank 112 --fedslop_momentum 0.8 `
  --sparsity 0.15 --select_ratio 0.15 --block_ratio 0.15 `
  --output data\mnist_alpha_sweep_methods.csv
```

FedLoRA-M:

```powershell
python code\experiment_mnist_fedlora_torch.py `
  --rounds 100 --num_clients 50 --alphas 0.05,0.1,0.5,1.0 `
  --local_epochs 1 --batch_size 32 --lr 0.018 `
  --rank 15 --seeds 0,1,2 --device cpu `
  --output data\mnist_alpha_sweep_fedlora.csv
```

### 3) Projection-rank ablation ($r$ sweep for FedSLoP)

```powershell
python scripts\run_rank_sweep.py `
  --ranks "16,32,64,112,192" `
  --rounds 100 --num_clients 50 --alpha 0.1 `
  --local_epochs 1 --batch_size 32 --lr 0.018 `
  --seeds "0,1,2" --device cpu --client_fraction 0.2 `
  --fedslop_momentum 0.8 `
  --output data\mnist_rank_sweep.csv
```

### 4) Scaling with number of clients ($p$ sweep for FedSLoP)

```powershell
python scripts\run_num_clients_sweep.py `
  --client_counts "10,20,50,100" `
  --rounds 100 --alpha 0.1 `
  --local_epochs 1 --batch_size 32 --lr 0.018 `
  --seeds "0,1,2" --device cpu --client_fraction 0.2 `
  --proj_rank 112 --fedslop_momentum 0.8 `
  --output data\mnist_num_clients_sweep.csv
```

## Citation

TODO

