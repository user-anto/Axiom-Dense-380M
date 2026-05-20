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
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC

def encode(text: str) -> list[int]:
    return get_tokenizer().encode_ordinary(text)

def decode(ids: list[int]) -> str:
    return get_tokenizer().decode(ids)

def get_eos_token_id() -> int:
    # cl100k_base exposes this as eot_token.
    return get_tokenizer().eot_token

VOCAB_SIZE = get_tokenizer().n_vocab  # 100277
