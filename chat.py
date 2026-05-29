#!/usr/bin/env python3
import argparse
import glob
import inspect
import os
import re
import time

import torch

from config import ModelConfig
from model import LLM
from tokenizer import encode, decode


SYSTEM_PROMPT = "You are a helpful and knowledgeable assistant. "

def latest_ckpt(ckpt_dir: str) -> str | None:
    paths = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not paths:
        return None
    paths.sort(key=lambda p: int(re.search(r"step_(\d+)\.pt$", os.path.basename(p)).group(1)))
    return paths[-1]


def load_model(ckpt_path: str):
    mcfg = ModelConfig()
    model = LLM(mcfg).to("cpu")
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    ckpt = torch.load(ckpt_path, **load_kwargs)
    state = ckpt.get("model", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="sft_checkpoints/step_0000641.pt")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--stream", action="store_true", help="stream tokens as they are generated")
    args = p.parse_args()

    ckpt_path = args.ckpt if args.ckpt else latest_ckpt(args.ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Use --ckpt or put checkpoints in --ckpt-dir.")

    print(f"loading {ckpt_path} on cpu...")
    model = load_model(ckpt_path)
    print("type /quit to exit")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        chatml_prompt = f"<|im_start|>user\n{SYSTEM_PROMPT + user}<|im_end|>\n<|im_start|>assistant\n"
        ids = encode(chatml_prompt)
        
        max_seq_len = model.cfg.max_seq_len
        
        if len(ids) > max_seq_len:
            ids = ids[-max_seq_len:]
            
        x = torch.tensor([ids], dtype=torch.long, device="cpu")
        stop_tokens = {100265, 100257}

        with torch.no_grad():
            if args.stream:
                print("Axiom> ", end="", flush=True)
                y = x
                prev_text = ""
                for _ in range(args.max_new_tokens):
                    y = model.generate(
                        y,
                        max_new_tokens=1,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                    )
                    last_tok = y[0, -1].item()
                    if last_tok in stop_tokens:
                        break
                    
                    out = y[0, len(ids) :].tolist()
                    text = decode(out)
                    if text.startswith(prev_text):
                        delta = text[len(prev_text) :]
                    else:
                        delta = text
                    if delta:
                        for char in delta:
                            print(char, end="", flush=True)
                            time.sleep(0.01)
                    prev_text = text
                print()
            else:
                y = model.generate(
                    x,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
                out = y[0, len(ids) :].tolist()
                for i, tok in enumerate(out):
                    if tok in stop_tokens:
                        out = out[:i]
                        break
                print("Axiom>", decode(out))


if __name__ == "__main__":
    main()
