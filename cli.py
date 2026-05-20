#!/usr/bin/env python3
"""
Ollama-style CLI inference for the AWQ-quantized LLM.

Usage:
    python cli.py --model models/MiniLM-awq                   # interactive chat
    python cli.py --model models/MiniLM-awq --prompt "Hello"  # single shot
    python cli.py --model models/MiniLM-awq --system "You are a helpful assistant."
"""
import argparse, sys, os, time
import torch

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
DIM    = "\033[2m"


def print_banner(model_path: str, device: str):
    print(f"\n{BOLD}{CYAN}┌─────────────────────────────────────────┐{RESET}")
    print(f"{BOLD}{CYAN}│   MiniLM  ·  AWQ 4-bit  ·  Inference   │{RESET}")
    print(f"{BOLD}{CYAN}└─────────────────────────────────────────┘{RESET}")
    print(f"{DIM}  model  : {model_path}{RESET}")
    print(f"{DIM}  device : {device}{RESET}")
    print(f"{DIM}  type /help for commands, Ctrl-C to quit{RESET}\n")


def load_model(model_path: str):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"{DIM}Loading model …{RESET}", end="\r")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model     = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=True,
        trust_remote_code=False,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model loaded on {device}          ")
    return model, tokenizer, str(device)


@torch.inference_mode()
def stream_generate(model, tokenizer, prompt: str, args) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
        repetition_penalty=args.rep_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=None,  # streaming via decode loop below
    )
    new_ids   = output[0][input_len:]
    response  = tokenizer.decode(new_ids, skip_special_tokens=True)
    elapsed   = time.time() - t0
    tok_per_s = len(new_ids) / elapsed
    print(f"\n{DIM}  ⏱  {len(new_ids)} tokens · {elapsed:.1f}s · {tok_per_s:.1f} tok/s{RESET}")
    return response


def build_prompt(history: list[dict], system: str) -> str:
    """Simple ChatML-style prompt (same format used by many open models)."""
    lines = []
    if system:
        lines.append(f"<|system|>\n{system}")
    for turn in history:
        role = turn["role"]
        lines.append(f"<|{role}|>\n{turn['content']}")
    lines.append("<|assistant|>")
    return "\n".join(lines)


HELP_TEXT = f"""
{BOLD}Commands:{RESET}
  /help          show this message
  /clear         clear conversation history
  /system <msg>  set system prompt
  /temp <float>  set temperature (current session)
  /tokens <int>  set max new tokens
  /quit          exit
"""


def interactive_loop(model, tokenizer, args):
    history: list[dict] = []
    system = args.system

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd, *rest = user_input.split(maxsplit=1)
            if cmd == "/quit":
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/clear":
                history = []
                print(f"{DIM}History cleared.{RESET}")
            elif cmd == "/system":
                system = rest[0] if rest else ""
                print(f"{DIM}System prompt set.{RESET}")
            elif cmd == "/temp":
                args.temperature = float(rest[0])
                print(f"{DIM}Temperature → {args.temperature}{RESET}")
            elif cmd == "/tokens":
                args.max_tokens = int(rest[0])
                print(f"{DIM}Max tokens → {args.max_tokens}{RESET}")
            continue

        history.append({"role": "user", "content": user_input})
        prompt   = build_prompt(history, system)
        response = stream_generate(model, tokenizer, prompt, args)
        history.append({"role": "assistant", "content": response})
        print(f"\n{BOLD}{CYAN}Assistant:{RESET} {response}\n")


def main():
    parser = argparse.ArgumentParser(description="MiniLM CLI Inference")
    parser.add_argument("--model",       default="models/MiniLM-awq")
    parser.add_argument("--prompt",      default=None,   help="Non-interactive single prompt")
    parser.add_argument("--system",      default="",     help="System prompt")
    parser.add_argument("--max-tokens",  type=int,   default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p",       type=float, default=0.9)
    parser.add_argument("--top-k",       type=int,   default=50)
    parser.add_argument("--rep-penalty", type=float, default=1.1)
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"Error: model directory '{args.model}' not found.", file=sys.stderr)
        print("Run `python quantize.py --ckpt <ckpt> --out models/MiniLM-awq` first.")
        sys.exit(1)

    model, tokenizer, device = load_model(args.model)
    print_banner(args.model, device)

    if args.prompt:
        prompt   = build_prompt([{"role": "user", "content": args.prompt}], args.system)
        response = stream_generate(model, tokenizer, prompt, args)
        print(response)
    else:
        interactive_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()