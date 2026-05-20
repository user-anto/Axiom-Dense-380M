#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_train_metrics(path: str):
    steps, train_loss, lr = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            lr.append(float(row["lr"]))
    return steps, train_loss, lr


def read_eval_metrics(path: str):
    steps, eval_loss = [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            eval_loss.append(float(row["eval_loss"]))
    return steps, eval_loss


def main():
    p = argparse.ArgumentParser(description="Plot train/eval loss and LR curves from CSV logs.")
    p.add_argument("--train-csv", type=str, default="train_metrics.csv")
    p.add_argument("--eval-csv", type=str, default="eval.csv")
    p.add_argument("--out", type=str, default="training_curves.png")
    p.add_argument("--dpi", type=int, default=170)
    args = p.parse_args()

    train_steps, train_loss, lr = read_train_metrics(args.train_csv)
    eval_steps, eval_loss = read_eval_metrics(args.eval_csv)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(train_steps, train_loss, color="#1f77b4", linewidth=1.8, label="train_loss")
    axes[0].set_title("Training Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(eval_steps, eval_loss, color="#d62728", linewidth=1.8, marker="o", markersize=3, label="eval_loss")
    axes[1].set_title("Evaluation Loss")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(train_steps, lr, color="#2ca02c", linewidth=1.8, label="learning_rate")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("LR")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.suptitle("Axiom Dense Training Curves", y=0.995, fontsize=14)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
