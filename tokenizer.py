"""
Thin wrapper around tiktoken's cl100k_base (GPT-4 BPE, 100k vocab).
If you prefer a 32k vocab, swap to a trained SentencePiece/HF tokenizer.
Remember to set ModelConfig.vocab_size to match.
"""
import tiktoken

_ENC = None

def get_tokenizer():
    global _ENC
    if _ENC is None:
        base = tiktoken.get_encoding("cl100k_base")
        special = base._special_tokens.copy()
        # Patch unused dummy tokens for ChatML to avoid fragmentation
        special["<|im_start|>"] = 100264
        special["<|im_end|>"] = 100265
        _ENC = tiktoken.Encoding(
            name="chatml_cl100k",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens=special
        )
    return _ENC

def encode(text: str, allowed_special: set | str = "all") -> list[int]:
    # Use encode instead of encode_ordinary to parse the patched special tokens
    return get_tokenizer().encode(text, allowed_special=allowed_special)

def decode(ids: list[int]) -> str:
    return get_tokenizer().decode(ids)

def get_eos_token_id() -> int:
    # cl100k_base exposes this as eot_token.
    return get_tokenizer().eot_token

VOCAB_SIZE = get_tokenizer().n_vocab  # 100277
