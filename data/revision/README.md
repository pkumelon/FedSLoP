# Matched experiment records

These records accompany the updated MNIST and CIFAR-10 benchmarks. Each formal method uses paired seeds 0, 1, and 2. Accuracy values in CSV files are fractions. The main README displays percentages.

## Files

- `paper_summary.csv` contains the nine rows reported in the matched comparison tables, including the source summary path for every row.
- `config.json` records the model, data partition sizes, hyperparameters, device, and the actual projection or SVD ranks.
- `metrics.csv` records the training round, evaluation split, accuracy, and per-round and cumulative communication. CIFAR-10 records also include training loss, evaluation loss, and elapsed time.
- `selected_lr.json` records the validation scores and selected initial learning rate.
- `done` marks a completed run.

Absolute data and output paths in archived configs have been replaced by repository-relative paths, and CSV line endings are normalized to LF. Numerical parameters, ranks, metrics, and device information are preserved. Missing optional SVD fields use the launcher defaults, `svd_fraction=0.05` and `svd_rank_rule=payload`.

## Run groups

| Directory | Contents |
|---|---|
| `mnist_10of50_matched/` | FedSLoP, FedAvg-M, FedSub-CD, and full-momentum FedMuon, with their validation screens |
| `mnist_fedmuon_svd_budget/` | FedMuon-SVD† with extra SVD of model increments, matched to the 62,376 B uplink budget |
| `mnist_fedmuon_svd_rank_probe/` | Momentum-only SVD using the 5% matrix-rank rule, ranks (6, 1) |
| `mnist_fedmuon_svd/` | Momentum-only SVD using the 5% factor-payload rule, ranks (5, 0) |
| `mnist_fedmuon_svd_aggressive/` | 2% and 1% matrix-rank retention, ranks (2, 1) and (1, 1) |
| `mnist_client_memory/` | Complete client-update CUDA allocation measurements and summary |
| `cifar10_formal_steps20_server_momentum/` | Formal ResNet-20 runs for FedSLoP, FedAvg-M, FedSub-CD, and FedLoRA-M |
| `cifar10_matched_steps20/tune/` | Their five-point validation learning-rate screens and selection records |
| `cifar10_fedmuon_svd_budget/` | FedMuon-SVD† matched to the 816,848 B uplink budget |
| `cifar10_fedmuon_svd/` | Full-increment FedMuon-SVD at identical momentum ranks to the budget-matched version |
| `cifar10_fedmuon_svd_aggressive/` | Consistent matrix-rank rule at 5% and 2%, the latter yielding rank one in every matrix |
| `cifar10_fedmuon_full/` | Full-momentum FedMuon and its validation screen |

The CIFAR-10 core-method validation records use the fields `validation_accuracy_by_lr`, `selected_lr`, and `nonfinite_lr`. The extension runs store per-method `validation_scores` and `lr`. All reported choices maximize the final validation accuracy on the dataset's shared grid.

The core CIFAR-10 summary uses `test_accuracy_mean`, `test_accuracy_sd`, `learning_rate`, and `total_*_bytes`; extension summaries use `accuracy_mean`, `accuracy_sd`, `lr`, and `*_bytes`. `paper_summary.csv` provides one consistent schema for the reported table rows.

## Implementation and communication

FedSLoP stores round-local momentum in the projected coordinates. Each client uploads its projected model increment and unprojected vector parameters. CIFAR-10 uses server momentum with coefficient 0.8 after reconstruction.

FedSub-CD uses signed coordinate projections and per-client coordinate dual variables. The MNIST implementation maintains one state per client across sampled rounds and refreshes the shared basis every 50 rounds. Its state update is in `code/mnist_matched.py`.

FedMuon-SVD uploads SVD factors of the client momentum. The † version additionally uploads SVD factors of matrix-valued model increments, while vector parameters and BatchNorm buffers retain their recorded representation. The server decodes every client increment before model averaging and uses the same decoded value for the global alignment direction. The original and † rows in each paired comparison have identical momentum ranks and training settings.

The actual SVD ranks are stored as `fedmuon_momentum_svd_ranks` and `fedmuon_delta_svd_ranks` in each configuration. The MNIST † increment ranks are (10, 5). CIFAR-10 chooses increment ranks from the total byte budget before training. The total uplink, relative to FedAvg-M, is 15.31% on MNIST and 75.32% on CIFAR-10.

Communication is the number of transmitted tensor values times their element sizes, summed over clients and rounds. The accounting includes the recorded shared seeds and BatchNorm buffers. The FedMuon variants download the global model, complete aggregate momentum, and global alignment direction. FedSub-CD also receives the average coordinate increment. FedLoRA-M downloads its frozen weights once and then receives the trainable factors and other changing tensors.

## Reproducing

Use `scripts/run_matched_benchmarks.py` and the commands in the main README. Fresh runs can be written to `data/reproduced/`. The updated runner's method directory is named `FedSLoP`. The archived CIFAR-10 directory `FedSLoP-rank75-server` identifies the same 75%-rank server-momentum configuration.
