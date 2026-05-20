#!/usr/bin/env python3
"""
Prepare and upload the axiom base checkpoint to Hugging Face Hub.

Examples:
  python upload.py --repo-id yourname/axiom-base --ckpt checkpoints/step_0024414.pt
  python upload.py --repo-id yourname/axiom-base --private
  python upload.py --repo-id yourname/axiom-base --local-only
"""

import argparse
import glob
import inspect
import json
import os
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

from config import ModelConfig
from tokenizer import VOCAB_SIZE, get_eos_token_id

CONFIGURATION_axiom_PY = """from transformers import PretrainedConfig


class axiomConfig(PretrainedConfig):
    model_type = "axiom"

    def __init__(
        self,
        vocab_size=100277,
        dim=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=2.6667,
        max_seq_len=1024,
        rope_theta=10000.0,
        norm_eps=1e-5,
        dropout=0.0,
        bos_token_id=None,
        eos_token_id=100257,
        pad_token_id=100257,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.dropout = dropout
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
"""

MODELING_axiom_PY = """import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_axiom import axiomConfig
from config import ModelConfig
from model import LLM


class axiomForCausalLM(PreTrainedModel):
    config_class = axiomConfig
    base_model_prefix = "model"
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, config: axiomConfig):
        super().__init__(config)
        core_cfg = ModelConfig(
            vocab_size=config.vocab_size,
            dim=config.dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            norm_eps=config.norm_eps,
            dropout=config.dropout,
        )
        self.model = LLM(core_cfg)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value
        self.model.lm_head.weight = value.weight

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if past_key_values is None:
            return past_key_values
        out = []
        for layer in past_key_values:
            if layer is None:
                out.append(layer)
                continue
            k, v = layer
            out.append((k.index_select(0, beam_idx), v.index_select(0, beam_idx)))
        return out

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if use_cache is None:
            use_cache = True
        logits, loss, new_cache = self.model(
            input_ids,
            targets=labels,
            cache=past_key_values,
            use_grad_ckpt=False,
            return_cache=bool(use_cache),
        )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_cache if use_cache else None,
        )
"""

TOKENIZATION_axiom_PY = """import json
import os
import tiktoken
from transformers import PreTrainedTokenizer


class axiomTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"tokenizer_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_file=None,
        encoding_name="cl100k_base",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<|unk|>",
        **kwargs,
    ):
        if tokenizer_file and os.path.isfile(tokenizer_file):
            with open(tokenizer_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            encoding_name = payload.get("encoding_name", encoding_name)
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(self.encoding_name)
        super().__init__(
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return int(self._enc.n_vocab)

    def get_vocab(self):
        return {f"<|{i}|>": i for i in range(self.vocab_size)}

    def _tokenize(self, text, **kwargs):
        ids = self._enc.encode_ordinary(text)
        return [f"<|{i}|>" for i in ids]

    def _convert_token_to_id(self, token):
        if token == self.eos_token or token == self.pad_token:
            return int(self._enc.eot_token)
        if token.startswith("<|") and token.endswith("|>"):
            n = token[2:-2]
            if n.isdigit():
                return int(n)
        return int(self._enc.eot_token)

    def _convert_id_to_token(self, index):
        return f"<|{int(index)}|>"

    def convert_tokens_to_string(self, tokens):
        ids = [self._convert_token_to_id(t) for t in tokens]
        return self._enc.decode(ids)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        out_name = "tokenizer.model" if filename_prefix is None else f"{filename_prefix}-tokenizer.model"
        out_path = os.path.join(save_directory, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"encoding_name": self.encoding_name}, f)
        return (out_path,)
"""


def latest_ckpt(ckpt_dir: str) -> str | None:
    paths = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not paths:
        return None
    paths.sort(key=lambda p: int(re.search(r"step_(\d+)\.pt$", os.path.basename(p)).group(1)))
    return paths[-1]


def checkpoint_step(path: str, ckpt_obj: dict) -> int:
    if isinstance(ckpt_obj, dict) and "step" in ckpt_obj:
        return int(ckpt_obj["step"])
    m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_checkpoint(path: str):
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def normalize_state_dict(state: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def safetensors_ready_state_dict(state: dict) -> dict:
    """
    safetensors does not allow shared-storage tensors (e.g., tied weights).
    Clone secondary tensors in each shared-storage group.
    """
    out = dict(state)
    groups = {}
    for name, tensor in out.items():
        ptr = tensor.untyped_storage().data_ptr()
        groups.setdefault(ptr, []).append(name)
    for names in groups.values():
        if len(names) <= 1:
            continue
        # Keep one tensor as-is, clone the rest to break shared storage.
        for tied_name in names[1:]:
            out[tied_name] = out[tied_name].clone()
    return out


def prepare_export_dir(ckpt_path: str, export_dir: Path, repo_id: str):
    export_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(ckpt_path)
    state = ckpt.get("model", ckpt)
    state = normalize_state_dict(state)
    state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    step = checkpoint_step(ckpt_path, ckpt if isinstance(ckpt, dict) else {})

    # Save both legacy and preferred HF weight formats.
    torch.save(state, export_dir / "pytorch_model.bin")
    save_file(safetensors_ready_state_dict(state), export_dir / "model.safetensors")

    eos_id = int(get_eos_token_id())
    mcfg = asdict(ModelConfig())
    mcfg["vocab_size"] = VOCAB_SIZE
    hf_config = {
        "architectures": ["axiomForCausalLM"],
        "model_type": "axiom",
        "auto_map": {
            "AutoConfig": "configuration_axiom.axiomConfig",
            "AutoModelForCausalLM": "modeling_axiom.axiomForCausalLM",
            "AutoTokenizer": "tokenization_axiom.axiomTokenizer",
        },
        "vocab_size": mcfg["vocab_size"],
        "dim": mcfg["dim"],
        "n_layers": mcfg["n_layers"],
        "n_heads": mcfg["n_heads"],
        "n_kv_heads": mcfg["n_kv_heads"],
        "ffn_dim_multiplier": mcfg["ffn_dim_multiplier"],
        "max_seq_len": mcfg["max_seq_len"],
        "rope_theta": mcfg["rope_theta"],
        "norm_eps": mcfg["norm_eps"],
        "dropout": mcfg["dropout"],
        "bos_token_id": None,
        "eos_token_id": eos_id,
        "pad_token_id": eos_id,
    }
    (export_dir / "config.json").write_text(json.dumps(hf_config, indent=2), encoding="utf-8")

    tokenizer_cfg = {
        "tokenizer_class": "axiomTokenizer",
        "auto_map": {"AutoTokenizer": "tokenization_axiom.axiomTokenizer"},
        "tokenizer_backend": "tiktoken",
        "encoding_name": "cl100k_base",
        "model_max_length": int(mcfg["max_seq_len"]),
        "vocab_size": VOCAB_SIZE,
        "eos_token": "<|endoftext|>",
        "eos_token_id": eos_id,
        "pad_token": "<|endoftext|>",
        "pad_token_id": eos_id,
        "unk_token": "<|unk|>",
    }
    (export_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_cfg, indent=2), encoding="utf-8"
    )
    (export_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|unk|>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (export_dir / "tokenizer.model").write_text(
        json.dumps({"encoding_name": "cl100k_base"}, indent=2),
        encoding="utf-8",
    )
    (export_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "eos_token_id": eos_id,
                "pad_token_id": eos_id,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "source_checkpoint": ckpt_path,
        "step": step,
        "has_optimizer_state": isinstance(ckpt, dict) and ("optimizer" in ckpt),
        "export_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    (export_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (export_dir / "configuration_axiom.py").write_text(CONFIGURATION_axiom_PY, encoding="utf-8")
    (export_dir / "modeling_axiom.py").write_text(MODELING_axiom_PY, encoding="utf-8")
    (export_dir / "tokenization_axiom.py").write_text(TOKENIZATION_axiom_PY, encoding="utf-8")
    (export_dir / "requirements.txt").write_text(
        "torch\ntransformers>=4.40.0\ntiktoken\nsafetensors\n",
        encoding="utf-8",
    )

    # Source files to instantiate and run the model.
    for fname in ("model.py", "config.py", "tokenizer.py", "chat.py"):
        src = Path(fname)
        if src.exists():
            shutil.copy2(src, export_dir / fname)

    return step


def push_to_hub(export_dir: Path, repo_id: str, private: bool, token: str | None):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(export_dir),
        commit_message="Upload Axiom-Dense-380M-Base changes.",
    )


def main():
    p = argparse.ArgumentParser(description="Prepare/upload axiom base checkpoint to Hugging Face Hub")
    p.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo id, e.g. username/axiom-base")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (.pt). Defaults to latest in --ckpt-dir.")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to scan when --ckpt not set.")
    p.add_argument("--export-dir", type=str, default="hf_export", help="Local export directory.")
    p.add_argument("--private", action="store_true", help="Create/upload as private repo.")
    p.add_argument("--token", type=str, default=None, help="HF token. If omitted, uses HF_TOKEN env or cached login.")
    p.add_argument("--local-only", action="store_true", help="Only prepare local export, do not upload.")
    args = p.parse_args()

    ckpt_path = args.ckpt if args.ckpt else latest_ckpt(args.ckpt_dir)

    export_dir = Path(args.export_dir).resolve()
    step = prepare_export_dir(ckpt_path, export_dir=export_dir, repo_id=args.repo_id)
    print(f"[ok] Export prepared at: {export_dir}")
    print(f"[ok] Checkpoint step: {step}")

    if args.local_only:
        print("[ok] --local-only enabled; skipping upload.")
        return

    token = args.token or os.getenv("HF_TOKEN")
    print("Starting upload...")
    push_to_hub(export_dir=export_dir, repo_id=args.repo_id, private=args.private, token=token)
    print(f"[ok] Uploaded to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
