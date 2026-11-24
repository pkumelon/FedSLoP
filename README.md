# FedSLoP

Paper: [*FedSLoP: Memory-Efficient Federated Learning with Low-Rank Gradient Projection*](https://arxiv.org/abs/2604.24012).

This repository provides minimal code to reproduce the experiments reported in the paper.

The [matched benchmarks](#matched-benchmarks) add the additional MNIST baselines and the CIFAR-10/ResNet-20 comparison.

## Introduction

Federated learning (FL) trains models across many clients without sharing raw data, but practical deployments must cope with data heterogeneity and strict resource constraints on edge devices. A recurring bottleneck is the client-side optimizer state (e.g., momentum buffers), which is full-dimensional for modern models and can dominate memory usage.

FedSLoP is a FedAvg-style federated optimization method that integrates **random low-rank subspace projections** with **momentum** to improve memory efficiency while staying compatible with the standard FL workflow. At each communication round, the algorithm samples a fresh low-dimensional (Stiefel) subspace and constrains local stochastic gradients and momentum updates to this subspace. This reduces the effective dimension of optimizer state and update directions **without modifying the underlying model architecture**.

**Main takeaways:**

- **Theory**: Under smoothness, bounded stochastic variance, and bounded client gradient dissimilarity, the full-participation analysis gives a leading stochastic term of order $O(1/\sqrt{N\tau T})$ at a fixed projection ratio, together with initialization and local-drift terms.
- **Accuracy under compression**: With a reported compression ratio of $1/7$, FedSLoP reaches 0.9511±0.0028 final test accuracy at $\alpha=0.1$ (T=100), close to FedAvg-M at 0.9551±0.0015, and higher than sparse baselines and FedLoRA-M (0.9247±0.0063).
- **Robustness to heterogeneity**: At $\alpha=0.05$, FedSLoP achieves 0.9269 final accuracy versus 0.9035 for FedAvg-M (≈2% absolute improvement).

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

If `torch/torchvision` installation fails on your machine (OS/CPU/GPU mismatch), follow the official PyTorch install guide to select a compatible build.

MNIST is downloaded automatically via `torchvision.datasets.MNIST(..., download=True)` into `data/torch_datasets/`.

## Reproducing the original MNIST experiments

### 1) Main comparison on non-IID MNIST ($\alpha=0.1$)

Run the baselines and FedSLoP (excluding FedLoRA-M):

```powershell
python code\experiment_mnist_federated_torch.py `
  --methods FedSLop,FedMef,FederatedSelect,NeuLite,FedAvg-M `
  --rounds 100 --num_clients 50 --alpha 0.1 `
  --local_epochs 1 --batch_size 32 --lr 0.018 `
  --client_fraction 0.2 --seeds 0,1,2 --device cpu `
  --proj_rank 112 --fedslop_momentum 0.9 `
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

## Matched benchmarks

[Experiment records](data/revision/README.md) include the configurations and metrics for the additional MNIST baselines and the CIFAR-10 comparison. The tables below report final test accuracy as mean ± sample standard deviation over seeds 0, 1, and 2.

### Additional MNIST baselines

The two-layer MLP uses 50 clients, 10 sampled per round, a label-wise Dirichlet split with α = 0.1, one local epoch, and batch size 32. All three methods run for 100 rounds with the original fixed learning rate (0.01839685275139246).

| Method | Accuracy (%) | Uplink (bytes per participating client per round) |
|---|---:|---:|
| FedMuon | 95.09 ± 0.18 | 813,608 |
| FedSub-CD | 89.94 ± 0.32 | 62,376 |
| FedMuon-SVD† | 91.51 ± 2.45 | 62,320 |

FedMuon-SVD† uses truncated SVD for both momentum and matrix-valued model increments. FedSub-CD updates the dual states of participating clients and reprojects all stored duals when the basis changes.

Run from the repository root. MNIST downloads automatically.

```bash
python scripts/run_matched_benchmarks.py --dataset mnist --data-root data/torch_datasets --output data/reproduced/mnist_original_baselines
```

### CIFAR-10 with ResNet-20

All ten clients participate in each of 200 rounds with 20 local steps, batch size 64, and a label-wise Dirichlet split with α = 0.3. The launcher selects each method's learning rate on the same validation set and uses the same server momentum for FedSLoP, FedAvg-M, and FedLoRA-M. Cumulative uplink is summed across participating clients and rounds; 1 GB = 10⁹ bytes.

| Method | Accuracy (%) | Uplink (GB) |
|---|---:|---:|
| FedAvg-M | 80.51 ± 2.27 | 2.169 |
| FedSub-CD | 65.84 ± 2.60 | 1.634 |
| FedLoRA-M | 76.33 ± 2.90 | 1.613 |
| FedSLoP | 78.07 ± 3.43 | 1.634 |

Prepare CIFAR-10 before running the benchmark. The launcher defaults to CUDA for CIFAR-10 and CPU for MNIST.

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data/torch_datasets', train=True, download=True); CIFAR10('data/torch_datasets', train=False, download=True)"
python scripts/run_matched_benchmarks.py --dataset cifar --data-root data/torch_datasets --output data/reproduced/cifar10
```

Use `--method` to run one method, or `--tune-only` to run CIFAR-10 validation only. A completed seed contains `config.json`, `metrics.csv`, and `done`. Repeating a command resumes completed cases.
