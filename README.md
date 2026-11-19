# FedSLoP

Paper: [*FedSLoP: Memory-Efficient Federated Learning with Low-Rank Gradient Projection*](https://arxiv.org/abs/2604.24012).

This repository provides minimal code to reproduce the experiments reported in the paper.

The [matched benchmarks](#matched-benchmarks) add controlled MNIST and CIFAR-10/ResNet-20 comparisons, communication accounting, and complete experiment records.

## Introduction

Federated learning (FL) trains models across many clients without sharing raw data, but practical deployments must cope with data heterogeneity and strict resource constraints on edge devices. A recurring bottleneck is the client-side optimizer state (e.g., momentum buffers), which is full-dimensional for modern models and can dominate memory usage.

FedSLoP is a FedAvg-style federated optimization method that integrates **random low-rank subspace projections** with **momentum** to improve memory efficiency while staying compatible with the standard FL workflow. At each communication round, the algorithm samples a fresh low-dimensional (Stiefel) subspace and constrains local stochastic gradients and momentum updates to this subspace. This reduces the effective dimension of optimizer state and update directions **without modifying the underlying model architecture**.

**Main takeaways:**

- **Theory**: Under smoothness, bounded stochastic variance, and bounded client gradient dissimilarity, the full-participation analysis gives a leading stochastic term of order $O(1/\sqrt{NT})$ at a fixed projection ratio, together with initialization and local-drift terms.
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

## Matched benchmarks

The updated implementations use round-local, low-dimensional client momentum for FedSLoP and transmit model increments in shared subspace coordinates. The CIFAR-10 configuration adds momentum on the server after reconstructing the mean increment. Methods share data partitions, initialization, participation, minibatches, and local training budgets within each seed. Each method selects its learning rate using the same validation grid.

The implementation files are in `code/`, the experiment launcher is in `scripts/`, and the saved configurations and metrics are in [`data/revision/`](data/revision/README.md). The original entry points above reproduce the earlier MNIST experiment setup. Their `uplink_elems` field counts the dense objects passed by that simulator. The matched entry points explicitly encode the projected upload and report transmitted tensor bytes, including the compressed FedSLoP model increment.

### Settings and results

| Setting | MNIST | CIFAR-10 |
|---|---|---|
| Model | MLP, 784–128–10 | CIFAR ResNet-20 |
| Clients per round / total | 10 / 50 | 10 / 10 |
| Label Dirichlet parameter | 0.1 | 0.3 |
| Formal rounds | 100 | 200 |
| Local budget per round | 1 epoch | 20 steps |
| Batch size | 32 | 64 |
| Client momentum coefficient | 0.9 | 0.8 |
| Validation rounds, seed 0 | 20 | 30 |
| Learning-rate grid | 0.00919843, 0.01839685, 0.03679371 | 0.015, 0.03, 0.06, 0.09, 0.12 |
| Formal seeds | 0, 1, 2 | 0, 1, 2 |

MNIST holds out 5,000 training examples for validation and uses all 60,000 training examples in formal runs. CIFAR-10 uses 45,000 training examples and a fixed 5,000-example validation set. Its formal learning rate is multiplied by 0.1 after rounds 120 and 170. Recorded runs used FP32 with PyTorch 2.6.0+cu124 on NVIDIA A800 GPUs.

Accuracy is the final-round test mean and sample standard deviation over the three seeds. Communication sums transmitted tensors over all participating clients in one run. One GB is $10^9$ bytes.

| Dataset | Method | Accuracy (%) | Uplink (GB) | Downlink (GB) |
|---|---|---:|---:|---:|
| MNIST | FedSLoP | 95.90 ± 0.31 | 0.062 | 0.407 |
| | FedAvg-M | 96.13 ± 0.27 | 0.407 | 0.407 |
| | FedSub-CD | 91.14 ± 0.43 | 0.062 | 0.469 |
| | FedMuon-SVD† | 89.58 ± 3.03 | 0.062 | 1.221 |
| CIFAR-10 | FedSLoP | 78.07 ± 3.43 | 1.634 | 2.169 |
| | FedAvg-M | 80.51 ± 2.27 | 2.169 | 2.169 |
| | FedSub-CD | 65.84 ± 2.60 | 1.634 | 3.775 |
| | FedLoRA-M | 76.33 ± 2.90 | 1.613 | 1.624 |
| | FedMuon-SVD† | 80.27 ± 3.13 | 1.634 | 6.474 |

† adds truncated SVD of model increments to FedMuon-SVD's momentum compression. It uploads 62,320 B per client per round on MNIST and 816,840 B on CIFAR-10, matching FedSLoP/FedSub-CD's 62,376 B and 816,848 B up to rank rounding. Each decoded increment is used for both model aggregation and the global alignment direction. FedMuon-SVD retains full model-increment transmission when `--svd-uplink-budget` is omitted. [FedMuon](https://arxiv.org/abs/2510.27403) and [FedSub](https://arxiv.org/abs/2509.05213) describe the corresponding optimization mechanisms.

### Run the paired comparisons

Run commands from the repository root. The launcher defaults to CUDA on one GPU. Add `--device cpu` for CPU execution, or `--cuda-devices 0 1` to schedule independent runs on two GPUs. New runs below write to `data/reproduced/`.

MNIST runs the FedSLoP, FedAvg-M, and FedSub-CD learning-rate screens followed by their formal seeds.

```bash
python scripts/run_matched_benchmarks.py --dataset mnist --data-root data/torch_datasets --output data/reproduced/mnist_matched
```

Run the matched-uplink FedMuon-SVD† separately.

```bash
python scripts/run_matched_benchmarks.py --dataset mnist --method FedMuon-SVD --svd-rank-rule matrix --svd-fraction 0.05 --svd-uplink-budget 62376 --data-root data/torch_datasets --output data/reproduced/mnist_fedmuon_svd_budget
```

MNIST downloads automatically. Prepare CIFAR-10 once.

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data/torch_datasets', train=True, download=True); CIFAR10('data/torch_datasets', train=False, download=True)"
```

The CIFAR-10 launcher runs FedSLoP, FedAvg-M, FedSub-CD, and FedLoRA-M using the same 20 local steps and learning-rate grid.

```bash
python scripts/run_matched_benchmarks.py --dataset cifar --data-root data/torch_datasets --output data/reproduced/cifar10_matched
python scripts/run_matched_benchmarks.py --dataset cifar --method FedMuon-SVD --svd-rank-rule payload --svd-fraction 0.05 --svd-uplink-budget 816848 --data-root data/torch_datasets --output data/reproduced/cifar10_fedmuon_svd_budget
```

Each output contains `tuning/`, `selected_lr.json`, the formal seed directories, and `summary.csv`. A completed seed contains `config.json`, `metrics.csv`, and `done`. Repeating the command resumes completed cases. `--tune-only` runs validation selection by itself. Use `--method` to select one baseline.

### Momentum compression comparisons

The saved runs also include full model-increment transmission at the same momentum ranks. The paired results are:

| Dataset | Model-increment transmission | Accuracy (%) | Uplink (GB) | Downlink (GB) |
|---|---|---:|---:|---:|
| MNIST | Full | 97.03 ± 0.32 | 0.430 | 1.221 |
| | SVD† | 89.58 ± 3.03 | 0.062 | 1.221 |
| CIFAR-10 | Full | 82.85 ± 3.42 | 2.240 | 6.474 |
| | SVD† | 80.27 ± 3.13 | 1.634 | 6.474 |

To reproduce the full-increment rows, run the corresponding command above with `--svd-uplink-budget` omitted and a separate output directory.

The `matrix` rule uses `max(1, floor(fraction * min(m, n)))`. MNIST uses this rule at 5%, 2%, and 1%, giving momentum ranks `(6, 1)`, `(2, 1)`, and `(1, 1)`. Their accuracies are 97.03 ± 0.32%, 96.48 ± 0.17%, and 96.42 ± 0.64%.

The `payload` rule follows the released FedMuon code, `floor(fraction * m * n / (m + n + 1))`, including integer rounding to zero for small matrices. The paired CIFAR-10 rows above use its 5% setting. A separate, consistently defined matrix-rank sweep gives 84.33 ± 2.09% at 5% and 83.99 ± 2.22% at rank one. On this ResNet-20, 2% and 1% under the `matrix` rule produce the same rank-one configuration.

Set `--svd-rank-rule matrix --svd-fraction 0.02` or `0.01`, with a distinct output path, to reproduce these lower-rank runs. The [saved-run guide](data/revision/README.md) maps every configuration to its metrics.

### Client memory and plotting

The MNIST memory measurement uses the same MLP, FP32, batch size 32, and one local epoch. Run each method in a fresh process for seeds 0, 1, and 2.

```bash
python code/profile_mnist_client_memory.py --data-root data/torch_datasets --method FedSLoP --seed 0
```

The other method choices are `FedAvg-M` and `Full-client-momentum`. Saved measurements are in `data/revision/mnist_client_memory/`.

Regenerate the CIFAR-10 figure from the saved records.

```bash
python scripts/plot_matched_results.py --results-root data/revision --output picture/cifar10_accuracy_communication.pdf
```

## Citation

```bibtex
@article{he2026fedslop,
  title={FedSLoP: Memory-Efficient Federated Learning with Low-Rank Gradient Projection},
  author={He, Yutong and Huang, Zhengyang and Geng, Jiahe},
  journal={arXiv preprint arXiv:2604.24012},
  year={2026}
}
```

