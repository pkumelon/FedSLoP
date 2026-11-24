# Benchmark records

[paper_summary.csv](paper_summary.csv) collects the three additional MNIST rows in Table II and the four CIFAR-10 rows in Table VI. Accuracy is recorded as a fraction. The standard deviation is the sample standard deviation over seeds 0, 1, and 2.

| Directory | Records |
|---|---|
| [mnist_original_baselines](mnist_original_baselines/) | FedSub-CD, FedMuon, and FedMuon-SVD† under the original MNIST setting, using the fixed learning rate 0.01839685275139246 |
| [cifar10_formal_steps20_server_momentum](cifar10_formal_steps20_server_momentum/) | FedSLoP, FedAvg-M, FedSub-CD, and FedLoRA-M on CIFAR-10 with ResNet-20 |
| [cifar10_matched_steps20/tune](cifar10_matched_steps20/tune/) | CIFAR-10 validation runs and the selected learning rates |

Formal seed directories contain the configuration, per-round metrics, and a completion marker. The archived CIFAR-10 folder named FedSLoP-rank75-server contains the FedSLoP runs reported in Table VI. For MNIST, the run directory named FedMuon-SVD corresponds to FedMuon-SVD† in the paper. Its momentum ranks are (6, 1) and its model-increment ranks are (10, 5).

See the [main README](../../README.md#matched-benchmarks) for the run commands.
