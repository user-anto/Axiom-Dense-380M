from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 100277         # cl100k_base (tiktoken); must match tokenizer.VOCAB_SIZE
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8             # GQA: fewer KV heads than Q heads
    ffn_dim_multiplier: float = 2.6667
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

@dataclass
class TrainConfig:
    # Data
    dataset_path: str = "data/fineweb-edu-10BT"  # local path after save_to_disk
    dataset_name_field: str = "text"
    target_tokens: int = 8_000_000_000
    val_fraction: float = 0.001
    split_seed: int = 1337
    use_packed_data: bool = True
    prepare_packed_data: bool = False

    # Batch / accumulation — tuned for RTX 4070 8 GB
    batch_size: int = 1
    grad_accum_steps: int = 320     # effective batch = 1 * 256 * 1024 = 262144 tokens/step
    max_seq_len: int = 1024
    num_workers: int = 2

    # Optimiser
    optimizer: str = "adamw8bit"        # "adamw" or "adamw8bit" (bitsandbytes)
    fused_adamw: bool = True
    lr: float = 3e-4
    lr_min: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 0              # computed from target_tokens at runtime
    decay_fraction: float = 0.1     # final 10% of training for decay (Chinchilla-style)

    # Evaluation
    eval_interval: int = 500        # eval every N steps
    eval_steps: int = 100             # number of batches for eval

    # Metrics logging
    metrics_csv_path: str = "train_metrics.csv"
    eval_csv_path: str = "eval.csv"
    metrics_interval: int = 500
    attention_entropy_probe_len: int = 256
    gpu_peak_tflops: float = 0.0      # set >0 for MFU; 0 disables MFU

    # Precision & performance
    dtype: str = "bfloat16"
    compile_model: bool = False        # re-enabled: speedup worth the VRAM
    grad_checkpointing: bool = True

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    ckpt_interval: int = 1000
    ckpt_keep_last: int = 3           # keep only last N checkpoints; <=0 keeps all
    log_interval: int = 10
    micro_log_interval: int = 32      # print progress inside grad accumulation
    save_rng_state: bool = True
    save_loader_state: bool = True

    # Dry-run: verify VRAM before committing to full training
    dry_run: bool = False
    dry_run_steps: int = 2
    
    # Reproducibility
    seed: int = 1337
