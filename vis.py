#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_train_metrics(path: str):
    steps, train_loss, lr = [], [], []
    prev_step = -1
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            step = int(row["step"])
            if step < prev_step:
                break
            prev_step = step
            steps.append(step)
            train_loss.append(float(row["train_loss"]))
            lr.append(float(row["lr"]))
    return steps, train_loss, lr


def read_eval_metrics(path: str):
    steps, eval_loss, perplexity = [], [], []
    prev_step = -1
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            step = int(row["step"])
            if step < prev_step:
                break
            prev_step = step
            steps.append(step)
            eval_loss.append(float(row["eval_loss"]))
            if "perplexity" in row and row["perplexity"]:
                perplexity.append(float(row["perplexity"]))
            elif "eval_perplexity" in row and row["eval_perplexity"]:
                perplexity.append(float(row["eval_perplexity"]))
    return steps, eval_loss, perplexity


def read_sft_metrics(path: str):
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    lr_steps, lr = [], []
    ppl_steps, perplexity = [], []

    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if "step" not in row or not row["step"]:
                continue
            step = int(row["step"])
            
            if "train_loss" in row and row["train_loss"]:
                train_steps.append(step)
                train_loss.append(float(row["train_loss"]))
                
            if "eval_loss" in row and row["eval_loss"]:
                eval_steps.append(step)
                eval_loss.append(float(row["eval_loss"]))
                
            if "lr" in row and row["lr"]:
                lr_steps.append(step)
                lr.append(float(row["lr"]))
                
            if "perplexity" in row and row["perplexity"]:
                ppl_steps.append(step)
                perplexity.append(float(row["perplexity"]))
                
    return train_steps, train_loss, eval_steps, eval_loss, lr_steps, lr, ppl_steps, perplexity


def main():
    p = argparse.ArgumentParser(description="Plot train/eval loss, LR, and perplexity curves from CSV logs.")
    p.add_argument("--train-csv", type=str, default="train_metrics.csv")
    p.add_argument("--eval-csv", type=str, default="eval.csv")
    p.add_argument("--sft-csv", type=str, default=None, help="Path to SFT_metrics.csv (if plotting SFT metrics)")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--dpi", type=int, default=170)
    args = p.parse_args()

    sft_mode = False
    sft_path = None
    if args.sft_csv:
        sft_mode = True
        sft_path = args.sft_csv
    elif not Path(args.train_csv).exists() and Path("SFT_metrics.csv").exists():
        sft_mode = True
        sft_path = "SFT_metrics.csv"

    if sft_mode:
        print(f"Reading SFT metrics from {sft_path}...")
        train_steps, train_loss, eval_steps, eval_loss, lr_steps, lr, ppl_steps, perplexity = read_sft_metrics(sft_path)
    else:
        print(f"Reading pretraining metrics from {args.train_csv} and {args.eval_csv}...")
        train_steps, train_loss, lr = read_train_metrics(args.train_csv)
        eval_steps, eval_loss, perplexity = read_eval_metrics(args.eval_csv)
        lr_steps = train_steps
        ppl_steps = eval_steps

        # Filter pretraining perplexity to start from step 500
        filtered_ppl = [(s, p) for s, p in zip(ppl_steps, perplexity) if s >= 500]
        if filtered_ppl:
            ppl_steps, perplexity = zip(*filtered_ppl)
            ppl_steps = list(ppl_steps)
            perplexity = list(perplexity)
        else:
            ppl_steps, perplexity = [], []

    prefix = "sft_" if sft_mode else ""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot Loss (Train & Eval)
    plt.figure(figsize=(11, 5))
    if train_loss:
        plt.plot(train_steps, train_loss, color="#1f77b4", linewidth=1.8, label="train_loss")
    if eval_loss:
        plt.plot(eval_steps, eval_loss, color="#d62728", linewidth=1.8, marker="o", markersize=3, label="eval_loss")
    plt.title("Training and Evaluation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    loss_out = out_dir / f"{prefix}loss.png"
    plt.savefig(loss_out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"saved: {loss_out}")

    # 2. Plot Learning Rate
    if lr:
        plt.figure(figsize=(11, 5))
        plt.plot(lr_steps, lr, color="#2ca02c", linewidth=1.8, label="learning_rate")
        plt.title("Learning Rate")
        plt.xlabel("Step")
        plt.ylabel("LR")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        lr_out = out_dir / f"{prefix}lr.png"
        plt.savefig(lr_out, dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"saved: {lr_out}")

    # 3. Plot Perplexity
    if perplexity:
        plt.figure(figsize=(11, 5))
        plt.plot(ppl_steps, perplexity, color="#9467bd", linewidth=1.8, marker="o", markersize=3, label="perplexity")
        plt.title("Perplexity")
        plt.xlabel("Step")
        plt.ylabel("Perplexity")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        ppl_out = out_dir / f"{prefix}perplexity.png"
        plt.savefig(ppl_out, dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"saved: {ppl_out}")



if __name__ == "__main__":
    main()
