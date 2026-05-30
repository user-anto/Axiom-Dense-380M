<p align="center">
  <img src="./axiom_logo.png" width="220" alt="Axiom Logo">
</p>

---

<h2 align="center">
Pretraining a 380M parameter Language Model on an RTX 4070
</h2>

<p align="center">
  <a href="https://huggingface.co/user-anto/Axiom-Dense-380M-Base"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Model"></a>
  <img src="https://img.shields.io/badge/Model-385.8M%20params-blue?style=for-the-badge" alt="Model Size">
  <img src="https://img.shields.io/badge/Context-1024-orange?style=for-the-badge" alt="Context Length">
  <img src="https://img.shields.io/badge/Tokenizer-cl100k__base-green?style=for-the-badge" alt="Tokenizer">
</p>

<div align="center">

- <a href="https://huggingface.co/user-anto/Axiom-Dense-380M-Base">https://huggingface.co/user-anto/Axiom-Dense-380M-Base</a>
- <a href="https://huggingface.co/user-anto/Axiom-Dense-380M-Instruct">https://huggingface.co/user-anto/Axiom-Dense-380M-Instruct</a>

</div>

<br>

## Axiom

This project is about building a language model from scratch, training it at scale on web text, debugging failure modes (like CUDA-OOM, overfitting, repetition loops), and packaging it for reproducible use.

**Axiom Dense 380M** is a full-stack training project:

- Decoder-only Transformer implementation in PyTorch
- Deterministic packed-data pipeline for long training runs
- Optimized training loop with resume logic, checkpoint retention, eval tracking, and telemetry
- Supervised Fine-tuning on conversational data
- Inference CLI and chat interface
- Training visualization and experiment tracking workflow

<br>

## What Was Built

### 1) Model Architecture (`model.py`)

- 24-layer dense decoder-only Transformer
- Hidden size 1024
- 16 attention heads with GQA (`n_kv_heads=8`)
- RoPE positional encoding
- RMSNorm + SwiGLU feedforward blocks
- Tied token embedding / output head weights

Parameter count: **385,849,344**.

### 2) Tokenization (`tokenizer.py`)

- `tiktoken` `cl100k_base` vocabulary
- Vocab size: **100,277**
- Explicit EOS handling in data pipeline

### 3) Pretraining Phase

## Data Pipeline (`data.py`)

- FineWeb-Edu dataset loading from local disk snapshot
- Optional one-time packing to `uint32` binary token files
- Deterministic train/val split via hash-based row assignment
- Resumable packed token loader for efficient long runs

## Training System (`train.py`)

- Gradient accumulation for effective large-token steps on constrained VRAM
- AdamW / AdamW8bit optimizer support
- Warmup + cosine decay schedule
- Automatic checkpointing and interrupt-safe saves
- Eval loop with perplexity tracking
- Metric CSV outputs for training diagnostics

### 4) Supervised Fine-tuning Phase

## Data Pipeline (`sft_data.py`)

- Uses Hugging Face `smoltalk` dataset for high-quality instruction following
- Formats conversations into ChatML format (`<|im_start|>`, `<|im_end|>`)
- **Right-pads** shorter conversations up to 1024 tokens to keep the `EOS` token effectively positioned within the context window
- **Applies strict loss masking** over user prompts and padding tokens, forcing the model to solely optimize the assistant responses
- Packs sequences into 1024-token contexts

## Fine-tuning System (`sft_train.py`)

- Similar to `train.py` but tuned for SFT dynamics

### 5) Inference (`chat.py`, `cli.py`)

- Interactive local generation for quick testing
- Streaming response mode in chat CLI
- Decoding controls for repetition mitigation

<br>

## Pre-training Snapshot

- Target training tokens: **8.0B**
- Effective tokens per optimizer step: **327,680**
- Planned optimizer steps: **24,414**
- Best recorded eval loss: **2.7394** (step 15,000)
- Best recorded eval perplexity: **15.4780** (step 15,000)
- Final logged eval loss: **2.8972** (step 24,000)
- Final logged eval perplexity: **18.1233** (step 24,000)

# Pre-training Curves

<p align="center">
  <img src="./figures/loss.png" width="80%" alt="Pretraining Loss">
  <br><br>
  <img src="./figures/lr.png" width="80%" alt="Pretraining Learning Rate">
  <br><br>
  <img src="./figures/perplexity.png" width="80%" alt="Pretraining Perplexity">
</p>

## Fine-tuning Snapshot

- Target training tokens: **~0.2B** (smoltalk dataset)
- Planned optimizer steps: **641**
- Best recorded eval loss: **1.2640** (step 630)
- Best recorded eval perplexity: **3.5397** (step 630)
- Final logged eval loss: **1.2867** (step 640)
- Final logged eval perplexity: **3.6210** (step 640)

# Fine-tuning Curves

<p align="center">
  <img src="./figures/sft_loss.png" width="80%" alt="SFT Loss">
  <br><br>
  <img src="./figures/sft_lr.png" width="80%" alt="SFT Learning Rate">
  <br><br>
  <img src="./figures/sft_perplexity.png" width="80%" alt="SFT Perplexity">
</p>

<br>

## Repository Layout

```text
.
├── model.py            # Transformer architecture
├── config.py           # Model + training hyperparameters
├── data.py             # Dataset loading, packing, and loaders
├── train.py            # Main pretraining script
├── sft_data.py         # Supervised fine-tuning data pipeline
├── sft_train.py        # Supervised fine-tuning training loop
├── tokenizer.py        # tiktoken wrapper (patches unused tokens for ChatML)
├── chat.py             # Interactive chat CLI
├── cli.py              # Base-checkpoint inference CLI
├── vis.py              # Train/eval/lr curve plotting utility
├── CLI.md              # CLI argument reference
├── figures/            # Generated training visualizations
├── train_metrics.csv   # Training telemetry history
├── SFT_metrics.csv     # Fine-tuning telemetry history
└── imp_ckpts/          # Training checkpoints
```

<br>

## Quick Start

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch datasets tiktoken python-dotenv safetensors huggingface_hub
```

### Train

```bash
python3 train.py
```

### Chat Locally

```bash
# Base model
python3 chat.py --ckpt checkpoints/step_0024414.pt --stream

# Instruct model (uses SFT checkpoint by default)
python3 chat.py --ckpt sft_checkpoints/step_0000641.pt
```

<br>

## Tooling Docs

- CLI options reference: [`CLI.md`](./CLI.md)
- Visualization utility: [`vis.py`](./vis.py)

<br>

## Hugging Face Usage

Base Model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "user-anto/Axiom-Dense-380M-Base"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
```

Instruct Model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "user-anto/Axiom-Dense-380M-Instruct"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
```

<br>

## Design Decisions and Tradeoffs

- **Why 380M scale?** Big enough to surface real pretraining dynamics, small enough to iterate independently.
- **Why packed token binaries?** Bypasses standard PyTorch `IterableDataset` Python overhead. Loading pre-compiled `uint32` token streams via `np.memmap` achieves extreme throughput and allows exact byte-offset tracking for fast, fault-tolerant resumption.
- **The 31st-Bit Target Mask for SFT:** To avoid doubling disk footprint with parallel label arrays, `sft_data.py` uses bitwise masking. Assistant tokens are bitwise-ORed with `0x80000000` during preprocessing. The dataloader uses this flag to compute the `-100` CrossEntropy target on the fly, maintaining the identical high-performance binary structure from pretraining.
- **Zero-Resizing Special Tokens:** The tokenizer patches standard ChatML boundaries (`<|im_start|>`, `<|im_end|>`) into unused dummy token slots (`100264`, `100265`) of the `cl100k_base` vocabulary. This preserves the exact vocab size (`100,277`), meaning the embedding matrix and language modeling head do not require resizing or surgical weight modifications for the SFT phase.
- **Stateless Inference with ChatML:** The `chat.py` CLI uses strict ChatML boundaries without left-padding. We found that left-padding caused generation failures because positional embeddings were trained exclusively on right-padded sequences.

<br>

## Current Limitations

- Context length capped at 1024 tokens
- Evaluation currently internal; broad external benchmarking is pending
- Decoding quality still sensitive to prompt and sampling settings

<br>

## Future Work

- Expand benchmark reporting
- Add longer-context variant
- Add recursive variant
- Improve inference and packaging
- Publish reproducible training/eval scripts for one-command replication

<br>
