"""FedSLop proj_rank sweep: run FedSLop with different proj_rank values."""
import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from experiment_mnist_federated_torch import get_mnist_datasets, run_federated_experiment


def run_rank_sweep(
    ranks,
    rounds=100,
    num_clients=50,
    alpha=0.1,
    local_epochs=1,
    batch_size=32,
    lr=0.018,
    seeds=(0,),
    device="cpu",
    client_fraction=0.2,
    fedslop_momentum=0.8,
    output="data/mnist_fedslop_rank_sweep.csv",
):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    train_ds, test_ds = get_mnist_datasets()
    rows = []
    for seed in seeds:
        for rank in ranks:
            print(f"Running FedSLop rank={rank}, seed={seed} ...")
            history, comm_stats = run_federated_experiment(
                method="FedSLop",
                train_ds=train_ds,
                test_ds=test_ds,
                num_clients=num_clients,
                alpha=alpha,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                device=device,
                client_fraction=client_fraction,
                proj_rank=rank,
                fedslop_momentum=fedslop_momentum,
            )
            for round_idx, acc in history:
                rows.append({
                    "method": "FedSLop",
                    "seed": seed,
                    "alpha": alpha,
                    "num_clients": num_clients,
                    "proj_rank": rank,
                    "round": round_idx,
                    "test_acc": acc,
                    "uplink_elems": comm_stats.get(round_idx, 0.0),
                })
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"Saved rank sweep results to {output}")

    # Also produce a summary CSV
    summary_path = output.replace("_sweep.csv", "_summary.csv")
    last_round = df["round"].max()
    summary = (
        df[df["round"] == last_round]
        .groupby("proj_rank")
        .agg(mean_acc=("test_acc", "mean"), std_acc=("test_acc", "std"),
             mean_uplink=("uplink_elems", "mean"))
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=str, default="16,32,64,112,192")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--client_fraction", type=float, default=0.2)
    parser.add_argument("--fedslop_momentum", type=float, default=0.8)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ranks = [int(x.strip()) for x in args.ranks.split(",")]
    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    run_rank_sweep(
        ranks=ranks,
        rounds=args.rounds,
        num_clients=args.num_clients,
        alpha=args.alpha,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seeds=seeds,
        device=args.device,
        client_fraction=args.client_fraction,
        fedslop_momentum=args.fedslop_momentum,
        output=args.output,
    )


if __name__ == "__main__":
    main()
