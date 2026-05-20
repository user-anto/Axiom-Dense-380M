import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_from_disk
from dotenv import load_dotenv

from config import TrainConfig
from tokenizer import encode, get_eos_token_id

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def _load_base_dataset(path: str):
    ds = load_from_disk(path)
    # Handle DatasetDict saved to disk.
    if hasattr(ds, "keys"):
        try:
            keys = list(ds.keys())
            if "train" in keys:
                return ds["train"]
            if len(keys) == 1:
                return ds[keys[0]]
        except Exception:
            pass
    return ds


def _is_val_example(global_idx: int, seed: int, val_fraction: float) -> bool:
    if val_fraction <= 0.0:
        return False
    if val_fraction >= 1.0:
        return True
    key = f"{seed}:{global_idx}".encode("utf-8")
    x = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return (x / (2**64)) < val_fraction


def _in_split(global_idx: int, split: str, seed: int, val_fraction: float) -> bool:
    is_val = _is_val_example(global_idx, seed, val_fraction)
    return is_val if split == "val" else (not is_val)


def packed_paths(cfg: TrainConfig) -> tuple[str, str, str]:
    root = cfg.dataset_path
    return (
        os.path.join(root, "train.packed.uint32.bin"),
        os.path.join(root, "val.packed.uint32.bin"),
        os.path.join(root, "packed_meta.json"),
    )


def prepare_packed_dataset(cfg: TrainConfig, force: bool = False):
    train_path, val_path, meta_path = packed_paths(cfg)
    if (
        not force
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(meta_path)
    ):
        return train_path, val_path, meta_path

    print("[data] Building packed token files (one-time)...")
    ds = _load_base_dataset(cfg.dataset_path)
    eos_id = get_eos_token_id()

    train_tmp = train_path + ".tmp"
    val_tmp = val_path + ".tmp"

    train_tokens = 0
    val_tokens = 0
    docs = 0

    with open(train_tmp, "wb") as f_train, open(val_tmp, "wb") as f_val:
        for row_idx, row in enumerate(ds):
            text = row[cfg.dataset_name_field]
            token_ids = encode(text)
            token_ids.append(eos_id)
            arr = np.asarray(token_ids, dtype=np.uint32)

            if _in_split(row_idx, "val", cfg.split_seed, cfg.val_fraction):
                arr.tofile(f_val)
                val_tokens += int(arr.size)
            else:
                arr.tofile(f_train)
                train_tokens += int(arr.size)

            docs += 1
            if docs % 50_000 == 0:
                print(
                    f"[data] docs={docs:,} | train_tokens={train_tokens:,} | val_tokens={val_tokens:,}"
                )

    os.replace(train_tmp, train_path)
    os.replace(val_tmp, val_path)

    meta = {
        "dtype": "uint32",
        "split_seed": cfg.split_seed,
        "val_fraction": cfg.val_fraction,
        "dataset_name_field": cfg.dataset_name_field,
        "eos_token_id": eos_id,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "docs": docs,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[data] packed ready: train={train_tokens:,} tokens, val={val_tokens:,} tokens"
    )
    return train_path, val_path, meta_path


class PackedTokenLoader:
    """
    Lightweight, resumable batch iterator over a packed uint32 token file.
    """

    def __init__(self, path: str, batch_size: int, seq_len: int, split: str, cycle: bool = True):
        self.path = path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1
        self.step_tokens = self.batch_size * self.chunk_size
        self.split = split
        self.cycle = cycle
        self.pos = 0
        self.epoch = 0

        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.n_tokens = int(self.tokens.shape[0])
        if self.n_tokens < self.step_tokens:
            raise ValueError(
                f"Packed split '{split}' too small for one batch: "
                f"need {self.step_tokens} tokens, found {self.n_tokens}"
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos + self.step_tokens > self.n_tokens:
            if not self.cycle:
                raise StopIteration
            self.pos = 0
            self.epoch += 1

        raw = np.asarray(
            self.tokens[self.pos : self.pos + self.step_tokens], dtype=np.int64
        )
        if raw.shape[0] < self.step_tokens:
            if not self.cycle:
                raise StopIteration
            self.pos = 0
            self.epoch += 1
            raw = np.asarray(
                self.tokens[self.pos : self.pos + self.step_tokens], dtype=np.int64
            )

        self.pos += self.step_tokens
        batch = raw.reshape(self.batch_size, self.chunk_size)
        x = torch.from_numpy(batch[:, :-1].copy())
        y = torch.from_numpy(batch[:, 1:].copy())
        return x, y

    def state_dict(self) -> dict:
        return {"pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, state: dict):
        if not state:
            return
        pos = int(state.get("pos", 0))
        pos = max(0, min(pos, max(0, self.n_tokens - self.step_tokens)))
        # Keep pointer aligned to full-batch boundaries.
        self.pos = (pos // self.step_tokens) * self.step_tokens
        self.epoch = int(state.get("epoch", 0))


class FineWebDataset(IterableDataset):
    def __init__(self, cfg: TrainConfig, split: str, worker_offset: int = 0, n_workers: int = 1):
        self.cfg = cfg
        self.split = split
        self.worker_offset = worker_offset
        self.n_workers = n_workers
        self.chunk_size = cfg.max_seq_len + 1

    def __iter__(self):
        ds = _load_base_dataset(self.cfg.dataset_path)
        ds = ds.to_iterable_dataset(num_shards=self.n_workers)
        ds = ds.shard(num_shards=self.n_workers, index=self.worker_offset)

        eos_id = get_eos_token_id()
        buf: list[int] = []
        ptr = 0

        for row_idx, row in enumerate(ds):
            global_idx = row_idx * self.n_workers + self.worker_offset
            if not _in_split(global_idx, self.split, self.cfg.split_seed, self.cfg.val_fraction):
                continue

            buf.extend(encode(row[self.cfg.dataset_name_field]))
            buf.append(eos_id)

            while ptr + self.chunk_size <= len(buf):
                chunk = buf[ptr : ptr + self.chunk_size]
                ptr += self.chunk_size
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )

            if ptr > 100_000:
                buf = buf[ptr:]
                ptr = 0


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.worker_offset = worker_id
    ds.n_workers = info.num_workers


def build_dataloader(
    cfg: TrainConfig,
    split: str = "train",
    num_workers: int = 4,
    persistent_workers: bool = True,
):
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    if cfg.use_packed_data:
        train_path, val_path, _ = packed_paths(cfg)
        if cfg.prepare_packed_data:
            prepare_packed_dataset(cfg)
        packed_path = train_path if split == "train" else val_path
        if not os.path.exists(packed_path):
            raise FileNotFoundError(
                f"Packed file missing: {packed_path}. Enable prepare_packed_data or build it first."
            )
        return PackedTokenLoader(
            path=packed_path,
            batch_size=cfg.batch_size,
            seq_len=cfg.max_seq_len,
            split=split,
            cycle=True,
        )

    persistent = persistent_workers and num_workers > 1
    return DataLoader(
        FineWebDataset(cfg, split=split),
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=persistent,
    )


def download_dataset(path: str = "data/fineweb-edu-10BT"):
    from datasets import load_dataset

    if os.path.exists(path):
        print(f"Dataset already exists at '{path}'. Skipping download.")
        return

    print("Downloading FineWeb-Edu sample-10BT (~30 GB). This runs once.")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        num_proc=1,
        token=HF_TOKEN,
    )
    os.makedirs(path, exist_ok=True)
    ds.save_to_disk(path)
    print(f"Done. {len(ds):,} documents saved to '{path}'.")


if __name__ == "__main__":
    download_dataset(path=TrainConfig().dataset_path)
