import json
import os
import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv

from config import SFTConfig
from tokenizer import encode, get_eos_token_id

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# High bit flag for targets (loss computed) vs prompts (loss ignored)
TARGET_FLAG = 0x80000000
TOKEN_MASK = 0x7FFFFFFF

def prepare_single_dataset(hf_path: str, out_dir: str, cfg: SFTConfig):
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(out_dir, "sft_train.packed.uint32.bin")
    val_path = os.path.join(out_dir, "sft_val.packed.uint32.bin")
    meta_path = os.path.join(out_dir, "sft_packed_meta.json")

    print(f"[sft_data] Loading dataset {hf_path}...")
    ds = load_dataset(
        hf_path,
        split="train",
        token=HF_TOKEN,
    )

    eos_id = get_eos_token_id()

    train_tmp = train_path + ".tmp"
    val_tmp = val_path + ".tmp"

    train_tokens = 0
    val_tokens = 0
    docs = 0

    print(f"[sft_data] Processing and packing tokens for {hf_path}...")

    with open(train_tmp, "wb") as f_train, open(val_tmp, "wb") as f_val:
        for row in ds:
            messages = row.get("messages", [])
            if not messages:
                continue

            buffer = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    continue
                
                # Format ChatML: <|im_start|>role\ncontent<|im_end|>\n
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                token_ids = encode(text)
                
                for tid in token_ids:
                    if role == "assistant":
                        # Assistant tokens are targets, apply bitmask flag
                        buffer.append(tid | TARGET_FLAG)
                    else:
                        # User/other tokens are prompts, no flag
                        buffer.append(tid)

            # Truncate if too long
            chunk_size = cfg.max_seq_len + 1
            if len(buffer) > chunk_size:
                buffer = buffer[:chunk_size]
                # Ensure the last token is EOS with target flag if we truncated
                buffer[-1] = eos_id | TARGET_FLAG
            
            # Pad with EOS (no target flag so loss ignores it) if too short
            while len(buffer) < chunk_size:
                buffer.append(eos_id)

            arr = np.asarray(buffer, dtype=np.uint32)
            
            # Split train/val deterministically
            if np.random.rand() < cfg.val_fraction:
                arr.tofile(f_val)
                val_tokens += int(arr.size)
            else:
                arr.tofile(f_train)
                train_tokens += int(arr.size)
                
            docs += 1
            if docs % 10_000 == 0:
                print(f"[sft_data] {hf_path} processed {docs:,} docs | train_tokens={train_tokens:,} | val_tokens={val_tokens:,}")
            
            if docs >= cfg.smoltalk_max_rows:
                print(f"[sft_data] Reached max rows ({cfg.smoltalk_max_rows}). Stopping.")
                break

    os.replace(train_tmp, train_path)
    os.replace(val_tmp, val_path)

    meta = {
        "dtype": "uint32",
        "val_fraction": cfg.val_fraction,
        "eos_token_id": eos_id,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "docs": docs,
        "target_flag": TARGET_FLAG,
        "token_mask": TOKEN_MASK,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[sft_data] {hf_path} SFT packed ready: train={train_tokens:,} tokens, val={val_tokens:,} tokens")
    return train_path, val_path, meta_path

class SFTPackedTokenLoader:
    """
    Lightweight, resumable batch iterator over a packed uint32 token file.
    Decodes the highest bit to determine loss masking targets.
    """
    def __init__(self, path: str, batch_size: int, seq_len: int, cycle: bool = True):
        self.path = path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1
        self.step_tokens = self.batch_size * self.chunk_size
        self.cycle = cycle
        self.pos = 0
        self.epoch = 0

        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.n_tokens = int(self.tokens.shape[0])
        if self.n_tokens < self.step_tokens:
            raise ValueError(f"Packed SFT dataset too small for one batch.")

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos + self.step_tokens > self.n_tokens:
            if not self.cycle:
                raise StopIteration
            self.pos = 0
            self.epoch += 1

        raw = np.asarray(
            self.tokens[self.pos : self.pos + self.step_tokens], dtype=np.uint32
        )
        if raw.shape[0] < self.step_tokens:
            if not self.cycle:
                raise StopIteration
            self.pos = 0
            self.epoch += 1
            raw = np.asarray(
                self.tokens[self.pos : self.pos + self.step_tokens], dtype=np.uint32
            )

        self.pos += self.step_tokens
        batch = raw.reshape(self.batch_size, self.chunk_size)
        
        x_raw = batch[:, :-1].copy()
        y_raw = batch[:, 1:].copy()
        
        x_tok = x_raw & TOKEN_MASK
        y_tok = y_raw & TOKEN_MASK
        is_target = (y_raw & TARGET_FLAG) != 0
        
        # if target predict y_tok, else predict -100
        y_final = np.where(is_target, y_tok.astype(np.int64), -100)
        x_final = x_tok.astype(np.int64)

        return torch.from_numpy(x_final), torch.from_numpy(y_final)

    def state_dict(self) -> dict:
        return {"pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, state: dict):
        if not state:
            return
        pos = int(state.get("pos", 0))
        pos = max(0, min(pos, max(0, self.n_tokens - self.step_tokens)))
        self.pos = (pos // self.step_tokens) * self.step_tokens
        self.epoch = int(state.get("epoch", 0))

if __name__ == "__main__":
    np.random.seed(1337)
    cfg = SFTConfig()
    
    prepare_single_dataset(
        "HuggingFaceTB/smol-smoltalk",
        "data/smol-smoltalk",
        cfg
    )
