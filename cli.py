#!/usr/bin/env python3
"""
Simple base-model CLI for Axiom checkpoints.

Examples:
  python3 cli.py --ckpt checkpoints/step_0024414.pt
  python3 cli.py --ckpt checkpoints/step_0024414.pt --prompt "Hello there"
"""

import argparse
import glob
import inspect
import os
import re

import torch

from config import ModelConfig
from model import LLM
from tokenizer import decode, encode


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


def build_prompt(history: list[dict], system: str) -> str:
    lines = []
    if system:
        lines.append(f"System: {system}")
    for turn in history:
        role = turn["role"]
        lines.append(f"{role.capitalize()}: {turn['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def generate_text(model, prompt: str, args) -> str:
    ids = encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device="cpu")
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    out_ids = y[0, len(ids) :].tolist()
    return decode(out_ids)


def interactive_loop(model, args):
    history: list[dict] = []
    system = args.system

    help_text = """
Commands:
  /help          show this message
  /clear         clear conversation history
  /system <msg>  set system prompt
  /temp <float>  set temperature
  /tokens <int>  set max new tokens
  /quit          exit
"""
    print("Type /help for commands.")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, *rest = user_input.split(maxsplit=1)
            if cmd == "/quit":
                break
            if cmd == "/help":
                print(help_text)
            elif cmd == "/clear":
                history = []
                print("history cleared")
            elif cmd == "/system":
                system = rest[0] if rest else ""
                print("system prompt set")
            elif cmd == "/temp":
                args.temperature = float(rest[0])
                print(f"temperature={args.temperature}")
            elif cmd == "/tokens":
                args.max_tokens = int(rest[0])
                print(f"max_tokens={args.max_tokens}")
            continue

        history.append({"role": "user", "content": user_input})
        prompt = build_prompt(history, system)
        response = generate_text(model, prompt, args)
        history.append({"role": "assistant", "content": response})
        print(f"bot> {response}")


def main():
    p = argparse.ArgumentParser(description="Axiom base checkpoint CLI")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--prompt", type=str, default=None, help="single-shot prompt")
    p.add_argument("--system", type=str, default="")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--rep-penalty", type=float, default=1.1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = p.parse_args()

    ckpt_path = args.ckpt if args.ckpt else latest_ckpt(args.ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Use --ckpt or put checkpoints in --ckpt-dir.")

    print(f"loading {ckpt_path} on cpu...")
    model = load_model(ckpt_path)

    if args.prompt:
        prompt = build_prompt([{"role": "user", "content": args.prompt}], args.system)
        print(generate_text(model, prompt, args))
        return

    interactive_loop(model, args)


if __name__ == "__main__":
    main()
