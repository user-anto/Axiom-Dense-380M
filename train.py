import argparse
import csv
import glob
import inspect
import math
import os
import random
import re
import time
from contextlib import nullcontext

import numpy as np
import torch

from config import ModelConfig, TrainConfig
from data import build_dataloader
from model import LLM


def get_lr(step: int, cfg: TrainConfig, decay_start_override: int | None = None) -> float:
    """Warmup -> constant -> cosine decay."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps

    decay_start = (
        int(decay_start_override)
        if decay_start_override is not None
        else int(cfg.max_steps * (1 - cfg.decay_fraction))
    )
    if step < decay_start:
        return cfg.lr
    if step >= cfg.max_steps:
        return cfg.lr_min

    progress = (step - decay_start) / max(1, (cfg.max_steps - decay_start))
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_recent_eval_points(
    path: str, max_points: int = 16, upto_step: int | None = None
) -> list[tuple[int, float]]:
    if not os.path.exists(path):
        return []
    points: list[tuple[int, float]] = []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    step = int(row["step"])
                    if upto_step is not None and step > upto_step:
                        continue
                    points.append((step, float(row["eval_loss"])))
                except Exception:
                    continue
    except Exception:
        return []
    return points[-max_points:]


def should_trigger_early_decay_from_csv(
    eval_csv_path: str, upto_step: int | None = None
) -> bool:
    eval_points = load_recent_eval_points(eval_csv_path, max_points=16, upto_step=upto_step)
    if len(eval_points) < 3:
        return False
    recent = eval_points[-3:]
    return recent[0][1] < recent[1][1] < recent[2][1]


def infer_first_decay_trigger_step_from_csv(
    eval_csv_path: str, upto_step: int | None = None
) -> int | None:
    points = load_recent_eval_points(eval_csv_path, max_points=1_000_000, upto_step=upto_step)
    if len(points) < 3:
        return None
    for i in range(2, len(points)):
        l0 = points[i - 2][1]
        l1 = points[i - 1][1]
        l2 = points[i][1]
        if l0 < l1 < l2:
            return int(points[i][0])
    return None


def infer_decay_start_from_observed_lr(step: int, observed_lr: float, cfg: TrainConfig) -> int | None:
    if observed_lr is None or not math.isfinite(observed_lr):
        return None
    if observed_lr >= (cfg.lr - 1e-12):
        return None
    
    best_d = None
    best_err = float("inf")
    upper = min(step, cfg.max_steps)
    for d in range(0, upper + 1):
        cand = get_lr(step, cfg, decay_start_override=d)
        err = abs(cand - observed_lr)
        if err < best_err:
            best_err = err
            best_d = d
    return best_d


def cleanup_old_checkpoints(ckpt_dir: str, keep_last: int):
    if keep_last <= 0:
        return
    files = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if len(files) > keep_last:
        for old in files[:-keep_last]:
            os.remove(old)
            print(f"[ckpt] deleted old -> {os.path.basename(old)}")


def _parse_ckpt_step(path: str):
    m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def list_checkpoints(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return []
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
        step = _parse_ckpt_step(p)
        if step is not None:
            out.append((step, p))
    out.sort(key=lambda x: x[0])
    return out


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _get_loader_state(loader):
    if loader is None or not hasattr(loader, "state_dict"):
        return None
    try:
        return loader.state_dict()
    except Exception:
        return None


def _set_loader_state(loader, state):
    if loader is None or state is None or not hasattr(loader, "load_state_dict"):
        return
    try:
        loader.load_state_dict(state)
    except Exception as e:
        print(f"[ckpt] warning: failed to restore loader state: {e}")


def _adapt_state_dict_for_model(model, state_dict: dict) -> dict:
    target_prefixed = any(k.startswith("_orig_mod.") for k in model.state_dict().keys())
    src_prefixed = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if target_prefixed == src_prefixed:
        return state_dict
    if src_prefixed and not target_prefixed:
        return {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    return {f"_orig_mod.{k}": v for k, v in state_dict.items()}


def save_ckpt(
    model,
    optimizer,
    step,
    loss,
    cfg: TrainConfig,
    train_loader=None,
    eval_loader=None,
    lr_state: dict | None = None,
    ckpt_dir: str = None,
    cleanup: bool = True,
    tag: str = "ckpt",
):
    out_dir = ckpt_dir or cfg.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step:07d}.pt")
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "train_config": vars(cfg),
    }
    if cfg.save_rng_state:
        ckpt["rng_state"] = _capture_rng_state()
    if cfg.save_loader_state:
        ckpt["loader_state"] = {
            "train": _get_loader_state(train_loader),
            "eval": _get_loader_state(eval_loader),
        }
    if lr_state is not None:
        ckpt["lr_state"] = lr_state
    torch.save(ckpt, path)
    print(f"[{tag}] saved -> {path}")
    if cleanup:
        cleanup_old_checkpoints(out_dir, cfg.ckpt_keep_last)


def important_ckpt_steps(total_steps: int) -> list[int]:
    if total_steps <= 0:
        return []
    return sorted({max(1, (total_steps * i) // 10) for i in range(1, 11)})


def resolve_resume_path(
    cfg: TrainConfig,
    resume_mode: str,
    resume_from: str = None,
    resume_step: int = None,
    strict_step: bool = False,
):
    if resume_from is not None and resume_step is not None:
        raise ValueError("Use either --resume-from or --resume-step, not both.")

    if resume_from is not None:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"--resume-from path not found: {resume_from}")
        return resume_from

    ckpts = list_checkpoints(cfg.ckpt_dir)
    if resume_step is not None:
        exact = [p for s, p in ckpts if s == resume_step]
        if exact:
            return exact[-1]
        if strict_step:
            raise FileNotFoundError(
                f"No checkpoint for step {resume_step} in {cfg.ckpt_dir}. "
                "Use a smaller ckpt_interval or disable --strict-step."
            )
        earlier = [(s, p) for s, p in ckpts if s <= resume_step]
        if earlier:
            chosen_step, chosen_path = earlier[-1]
            print(
                f"[ckpt] requested step {resume_step}, using nearest earlier checkpoint step {chosen_step}"
            )
            return chosen_path
        raise FileNotFoundError(
            f"No checkpoint at or before step {resume_step} in {cfg.ckpt_dir}."
        )

    if resume_mode == "none":
        return None

    if not ckpts:
        return None
    return ckpts[-1][1]


def load_ckpt(
    path: str,
    model,
    optimizer,
    cfg: TrainConfig,
    train_loader=None,
    eval_loader=None,
    reset_optimizer: bool = False,
) -> tuple[int, dict | None]:
    if path is None:
        return 0, None

    load_kwargs = {"map_location": "cpu"}

    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    ckpt = torch.load(path, **load_kwargs)

    model_state = ckpt.get("model", ckpt)
    model_state = _adapt_state_dict_for_model(model, model_state)
    model.load_state_dict(model_state)

    if not reset_optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[ckpt] warning: optimizer state not restored: {e}")
    elif reset_optimizer:
        print("[ckpt] optimizer state reset by request (--reset-optimizer).")

    if cfg.save_rng_state and "rng_state" in ckpt:
        _restore_rng_state(ckpt["rng_state"])

    if cfg.save_loader_state and "loader_state" in ckpt:
        _set_loader_state(train_loader, ckpt["loader_state"].get("train"))
        _set_loader_state(eval_loader, ckpt["loader_state"].get("eval"))

    step = int(ckpt.get("step", _parse_ckpt_step(path) or 0))
    print(f"[ckpt] resumed from step {step} ({path})")
    lr_state = ckpt.get("lr_state") if isinstance(ckpt, dict) else None
    if not isinstance(lr_state, dict):
        lr_state = None
    return step, lr_state


def report_vram(step: int):
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  [VRAM] step {step}: allocated {alloc:.2f} GB | reserved {reserved:.2f} GB")


def current_vram_stats() -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        return float("nan"), float("nan"), float("nan")
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return alloc, reserved, peak


def init_metrics_csv(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "train_loss",
                "grad_norm",
                "lr",
                "update_to_weight_ratio",
                "mfu",
                "vram_allocated_gb",
                "vram_reserved_gb",
                "vram_peak_allocated_gb",
                "attention_entropy",
            ]
        )


def append_metrics_row(path: str, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                row.get("step"),
                row.get("train_loss"),
                row.get("grad_norm"),
                row.get("lr"),
                row.get("update_to_weight_ratio"),
                row.get("mfu"),
                row.get("vram_allocated_gb"),
                row.get("vram_reserved_gb"),
                row.get("vram_peak_allocated_gb"),
                row.get("attention_entropy"),
            ]
        )


def init_eval_csv(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "eval_loss", "eval_perplexity"])


def append_eval_row(path: str, step: int, eval_loss: float, eval_perplexity: float):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([step, eval_loss, eval_perplexity])


def model_weight_l2_norm(model) -> float:
    sq = 0.0
    with torch.no_grad():
        for p in model.parameters():
            sq += float((p.detach().float().pow(2).sum()).item())
    return math.sqrt(max(sq, 1e-30))


def estimate_update_to_weight_ratio(grad_norm: float, lr: float, weight_norm: float, grad_clip: float) -> float:
    g = float(grad_norm)
    if grad_clip > 0:
        g = min(g, grad_clip)
    return (lr * g) / max(weight_norm, 1e-30)


def infer_peak_tflops(cfg: TrainConfig) -> float:
    if cfg.gpu_peak_tflops > 0:
        return cfg.gpu_peak_tflops
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(0).lower()
    # Conservative defaults for common local cards.
    if "4070" in name:
        return 29.1
    if "4090" in name:
        return 82.6
    if "a100" in name:
        return 312.0
    return 0.0


def estimate_mfu(tok_per_sec: float, param_count: int, peak_tflops: float) -> float:
    if peak_tflops <= 0:
        return float("nan")
    # Rough transformer training estimate: ~6 * N FLOPs per token.
    achieved = 6.0 * float(param_count) * float(tok_per_sec) / 1e12
    return achieved / peak_tflops


@torch.no_grad()
def run_eval(model, eval_loader, eval_iter, cfg: TrainConfig, device: str, ctx, steps: int = 100):
    model.eval()
    loss_accum = 0.0
    batches = 0

    for _ in range(steps):
        try:
            x, y = next(eval_iter)
        except StopIteration:
            eval_iter = iter(eval_loader)
            try:
                x, y = next(eval_iter)
            except StopIteration:
                break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with ctx:
            _, loss, _ = model(x, targets=y, use_grad_ckpt=False, return_cache=False)
        loss_accum += loss.item()
        batches += 1

    model.train()
    if batches == 0:
        return float("nan"), float("inf"), eval_iter, 0
    avg_loss = loss_accum / batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, eval_iter, batches


def build_optimizer(model, cfg: TrainConfig, device: str):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if cfg.optimizer.lower() == "adamw8bit":
        try:
            import bitsandbytes as bnb

            print("[opt] Using bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(
                groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
            )
        except Exception as e:
            print(f"[opt] warning: AdamW8bit unavailable ({e}); falling back to AdamW.")

    adamw_kwargs = {}
    if cfg.fused_adamw and device == "cuda":
        try:
            if "fused" in inspect.signature(torch.optim.AdamW).parameters:
                adamw_kwargs["fused"] = True
                print("[opt] Using fused AdamW")
        except Exception:
            pass

    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        **adamw_kwargs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a few steps and exit")
    parser.add_argument(
        "--resume",
        choices=["latest", "none"],
        default="latest",
        help="Resume policy when no explicit checkpoint is provided.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a specific checkpoint file to resume from.",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Resume from checkpoint step N (exact or nearest earlier step).",
    )
    parser.add_argument(
        "--strict-step",
        action="store_true",
        help="With --resume-step, require exact step checkpoint.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load model weights but do not restore optimizer state.",
    )
    parser.add_argument(
        "--save-on-interrupt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save an emergency checkpoint on Ctrl-C (default: enabled).",
    )
    parser.add_argument("--ckpt-interval", type=int, default=None, help="Override checkpoint interval.")
    parser.add_argument("--ckpt-keep-last", type=int, default=None, help="Override retention; <=0 keeps all.")
    args = parser.parse_args()

    tcfg = TrainConfig()
    mcfg = ModelConfig()
    if args.dry_run:
        tcfg.dry_run = True
    if args.ckpt_interval is not None:
        tcfg.ckpt_interval = args.ckpt_interval
    if args.ckpt_keep_last is not None:
        tcfg.ckpt_keep_last = args.ckpt_keep_last
    if tcfg.ckpt_interval <= 0:
        raise ValueError("ckpt_interval must be > 0")

    if mcfg.max_seq_len != tcfg.max_seq_len:
        raise ValueError(
            f"Sequence length mismatch: ModelConfig.max_seq_len={mcfg.max_seq_len} "
            f"!= TrainConfig.max_seq_len={tcfg.max_seq_len}"
        )

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    random.seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokens_per_step = tcfg.batch_size * tcfg.max_seq_len * tcfg.grad_accum_steps
    tcfg.max_steps = tcfg.target_tokens // tokens_per_step
    planned_tokens = tcfg.max_steps * tokens_per_step
    dropped_tokens = tcfg.target_tokens - planned_tokens
    print(f"Target tokens      : {tcfg.target_tokens/1e9:.3f}B")
    print(f"Planned tokens     : {planned_tokens/1e9:.3f}B")
    print(f"Dropped by rounding: {dropped_tokens:,}")
    print(f"Optimizer steps    : {tcfg.max_steps}")
    print(f"Effective batch    : {tokens_per_step/1e3:.0f}k tokens/step")
    keep_desc = "all" if tcfg.ckpt_keep_last <= 0 else str(tcfg.ckpt_keep_last)
    print(f"Checkpoint policy  : every {tcfg.ckpt_interval} steps, keep {keep_desc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tcfg.dtype == "bfloat16":
        ptdtype = torch.bfloat16
    elif tcfg.dtype == "float16":
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        if device == "cuda" and ptdtype != torch.float32
        else nullcontext()
    )

    model = LLM(mcfg).to(device)
    param_count = model.param_count()
    peak_tflops = infer_peak_tflops(tcfg)
    print(f"Params             : {param_count/1e6:.1f}M")
    if peak_tflops > 0:
        print(f"MFU peak TFLOPS    : {peak_tflops:.1f}")
    init_metrics_csv(tcfg.metrics_csv_path)
    init_eval_csv(tcfg.eval_csv_path)

    if tcfg.compile_model and hasattr(torch, "compile"):
        print("[info] Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = build_optimizer(model, tcfg, device)

    if tcfg.use_packed_data:
        num_workers = 0
        persistent_workers = False
    else:
        num_workers = tcfg.num_workers
        persistent_workers = True

    train_loader = build_dataloader(
        tcfg,
        split="train",
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    train_iter = iter(train_loader)

    eval_loader = None
    eval_iter = None
    if not tcfg.dry_run:
        eval_loader = build_dataloader(
            tcfg,
            split="val",
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        eval_iter = iter(eval_loader)

    resume_path = resolve_resume_path(
        tcfg,
        resume_mode=args.resume,
        resume_from=args.resume_from,
        resume_step=args.resume_step,
        strict_step=args.strict_step,
    )
    if resume_path is None:
        print("[ckpt] starting from scratch (no checkpoint loaded).")
    start_step, resumed_lr_state = load_ckpt(
        resume_path,
        model,
        optimizer,
        tcfg,
        train_loader=train_loader,
        eval_loader=eval_loader,
        reset_optimizer=args.reset_optimizer,
    )

    if tcfg.wandb_enabled:
        import wandb

        wandb.init(project=tcfg.wandb_project, config={**vars(mcfg), **vars(tcfg)})

    resume_lr_hint = None
    if optimizer.param_groups:
        try:
            resume_lr_hint = float(optimizer.param_groups[0].get("lr"))
        except Exception:
            resume_lr_hint = None

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    max_steps = start_step + tcfg.dry_run_steps if tcfg.dry_run else tcfg.max_steps
    imp_ckpt_dir = "imp_ckpts"
    imp_steps = set(important_ckpt_steps(max_steps))
    if imp_steps and not tcfg.dry_run:
        print(f"Important ckpts    : {sorted(imp_steps)} -> {imp_ckpt_dir}/")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    loss_accum = float("nan")
    last_completed_step = start_step
    interrupted = False
    default_decay_start_step = int(tcfg.max_steps * (1 - tcfg.decay_fraction))
    decay_start_step = default_decay_start_step
    early_decay_armed = True
    if resumed_lr_state is not None:
        if "decay_start_step" in resumed_lr_state:
            try:
                decay_start_step = int(resumed_lr_state["decay_start_step"])
            except Exception:
                decay_start_step = default_decay_start_step
        decay_start_step = max(0, min(decay_start_step, tcfg.max_steps))
        if bool(resumed_lr_state.get("early_decay_triggered", False)):
            early_decay_armed = False
            print(f"[lr] restored early-decay state from checkpoint (start={decay_start_step}).")
    elif not tcfg.dry_run:
        inferred_decay_start = infer_decay_start_from_observed_lr(
            start_step, resume_lr_hint, tcfg
        )
        if inferred_decay_start is not None:
            decay_start_step = min(decay_start_step, inferred_decay_start)
            early_decay_armed = False
            print(
                f"[lr] restored decay timeline from checkpoint optimizer LR "
                f"(start={inferred_decay_start})."
            )
        else:
            inferred_trigger_step = infer_first_decay_trigger_step_from_csv(
                tcfg.eval_csv_path, upto_step=start_step
            )
            if inferred_trigger_step is not None:
                decay_start_step = min(decay_start_step, inferred_trigger_step)
                early_decay_armed = False
                print(
                    f"[lr] restored early-decay timeline from eval.csv "
                    f"(trigger step {inferred_trigger_step})."
                )
            elif should_trigger_early_decay_from_csv(tcfg.eval_csv_path, upto_step=start_step):
                decay_start_step = min(decay_start_step, start_step)
                early_decay_armed = False
                print(
                    f"[lr] early decay pre-triggered at resume step {start_step} "
                    f"from eval.csv trend."
                )
    try:
        for step in range(start_step, max_steps):
            step_start_t = time.time()
            lr = get_lr(step, tcfg, decay_start_override=decay_start_step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            loss_accum = 0.0
            for micro_step in range(tcfg.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with ctx:
                    _, loss, _ = model(
                        x,
                        targets=y,
                        use_grad_ckpt=tcfg.grad_checkpointing,
                        return_cache=False,
                    )
                    loss = loss / tcfg.grad_accum_steps

                loss.backward()
                loss_accum += loss.item()

                if tcfg.micro_log_interval > 0:
                    is_last_micro = (micro_step + 1) == tcfg.grad_accum_steps
                    if ((micro_step + 1) % tcfg.micro_log_interval == 0) or is_last_micro:
                        pct = 100.0 * (micro_step + 1) / tcfg.grad_accum_steps
                        print(
                            f"  [accum] step {step+1}/{max_steps} | "
                            f"micro {micro_step+1}/{tcfg.grad_accum_steps} ({pct:.1f}%)",
                            end="\r" if not is_last_micro else "\n",
                            flush=True,
                        )

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            last_completed_step = step + 1
            total_tokens = last_completed_step * tokens_per_step
            step_dt = time.time() - step_start_t
            step_tok_per_sec = tokens_per_step / max(step_dt, 1e-9)

            if (step + 1) % tcfg.log_interval == 0:
                dt = time.time() - t0
                tok_per_sec = tokens_per_step * tcfg.log_interval / max(dt, 1e-9)
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                print(
                    f"step {step+1:7d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                    f"grad_norm {gn:.3f} | {tok_per_sec/1e3:.1f}k tok/s | "
                    f"tokens {tokens_per_step:,} | total {total_tokens:,}"
                )
                if tcfg.wandb_enabled:
                    import wandb

                    wandb.log(
                        {
                            "loss": loss_accum,
                            "lr": lr,
                            "tok_per_sec": tok_per_sec,
                            "grad_norm": gn,
                        },
                        step=step + 1,
                    )
                t0 = time.time()
            elif tcfg.log_interval > 1:
                # Keep a lightweight step heartbeat even between full log intervals.
                steps_left = max_steps - (step + 1)
                eta_sec = step_dt * steps_left
                print(
                    f"step {step+1:7d}/{max_steps} | step_time {step_dt:.1f}s | lr {lr:.8f} | "
                    f"ETA {eta_sec/3600:.2f}h | tokens {tokens_per_step:,} | total {total_tokens:,}"
                )

            if device == "cuda" and (tcfg.dry_run or (step + 1) % tcfg.log_interval == 0):
                report_vram(step + 1)

            row = None
            if (step + 1) % tcfg.metrics_interval == 0:
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                weight_norm = model_weight_l2_norm(model)
                update_ratio = estimate_update_to_weight_ratio(gn, lr, weight_norm, tcfg.grad_clip)
                alloc_gb, reserved_gb, peak_alloc_gb = current_vram_stats()
                attn_entropy = float("nan")
                if hasattr(model, "probe_attention_entropy"):
                    try:
                        attn_entropy = model.probe_attention_entropy(
                            x.detach(), max_probe_len=tcfg.attention_entropy_probe_len
                        )
                    except Exception as e:
                        print(f"[metrics] warning: attention entropy probe failed: {e}")
                row = {
                    "step": step + 1,
                    "train_loss": loss_accum,
                    "grad_norm": gn,
                    "lr": lr,
                    "update_to_weight_ratio": update_ratio,
                    "mfu": estimate_mfu(step_tok_per_sec, param_count, peak_tflops),
                    "vram_allocated_gb": alloc_gb,
                    "vram_reserved_gb": reserved_gb,
                    "vram_peak_allocated_gb": peak_alloc_gb,
                    "attention_entropy": attn_entropy,
                }

            if not tcfg.dry_run and (step + 1) % tcfg.eval_interval == 0:
                eval_loss, ppl, eval_iter, eval_batches = run_eval(
                    model, eval_loader, eval_iter, tcfg, device, ctx, tcfg.eval_steps
                )
                print(
                    f"  [eval] step {step+1} | loss {eval_loss:.4f} | "
                    f"perplexity {ppl:.2f} | batches {eval_batches}"
                )
                if tcfg.wandb_enabled:
                    import wandb

                    wandb.log({"eval_loss": eval_loss, "perplexity": ppl}, step=step + 1)
                append_eval_row(tcfg.eval_csv_path, step + 1, eval_loss, ppl)
                if math.isfinite(eval_loss):
                    if early_decay_armed and should_trigger_early_decay_from_csv(
                        tcfg.eval_csv_path
                    ):
                        decay_start_step = min(decay_start_step, step + 1)
                        early_decay_armed = False
                        print(
                            f"[lr] early decay triggered at step {step+1} "
                            f"after 2 consecutive eval-loss increases."
                        )

            if row is not None:
                append_metrics_row(tcfg.metrics_csv_path, row)

            if (step + 1) in imp_steps and not tcfg.dry_run:
                imp_path = os.path.join(imp_ckpt_dir, f"step_{step+1:07d}.pt")
                if os.path.exists(imp_path):
                    print(f"[imp_ckpt] exists -> {imp_path}")
                else:
                    save_ckpt(
                        model,
                        optimizer,
                        step + 1,
                        loss_accum,
                        tcfg,
                        train_loader=train_loader,
                        eval_loader=eval_loader,
                        lr_state={
                            "decay_start_step": decay_start_step,
                            "early_decay_triggered": (not early_decay_armed),
                        },
                        ckpt_dir=imp_ckpt_dir,
                        cleanup=False,
                        tag="imp_ckpt",
                    )

            if (step + 1) % tcfg.ckpt_interval == 0 and not tcfg.dry_run:
                save_ckpt(
                    model,
                    optimizer,
                    step + 1,
                    loss_accum,
                    tcfg,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    lr_state={
                        "decay_start_step": decay_start_step,
                        "early_decay_triggered": (not early_decay_armed),
                    },
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupt] Caught Ctrl-C.")
        if not tcfg.dry_run and args.save_on_interrupt:
            interrupt_step = max(0, last_completed_step)
            print(f"[interrupt] Saving emergency checkpoint at step {interrupt_step}...")
            save_ckpt(
                model,
                optimizer,
                interrupt_step,
                loss_accum,
                tcfg,
                train_loader=train_loader,
                eval_loader=eval_loader,
                lr_state={
                    "decay_start_step": decay_start_step,
                    "early_decay_triggered": (not early_decay_armed),
                },
            )
        elif not tcfg.dry_run:
            print("[interrupt] Emergency checkpoint disabled by --no-save-on-interrupt.")

    trained_tokens = last_completed_step * tokens_per_step

    if tcfg.dry_run:
        print("\n--- Dry-run complete ---")
        if device == "cuda":
            report_vram(last_completed_step)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak VRAM allocated: {peak:.2f} GB")
        else:
            print("  CUDA not available; VRAM metrics are unavailable on CPU.")
        print(f"  Tokens represented by current step: {trained_tokens:,}")
        return

    if interrupted:
        print(f"Training paused at step {last_completed_step}. Tokens represented: {trained_tokens:,}")
        return

    save_ckpt(
        model,
        optimizer,
        tcfg.max_steps,
        loss_accum,
        tcfg,
        train_loader=train_loader,
        eval_loader=eval_loader,
        lr_state={
            "decay_start_step": decay_start_step,
            "early_decay_triggered": (not early_decay_armed),
        },
    )
    print(f"Training complete. Tokens represented by final step: {trained_tokens:,}")


if __name__ == "__main__":
    main()
