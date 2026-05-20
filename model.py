import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional
from config import ModelConfig


# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# RoPE
def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, T, n_heads, head_dim)
    x_even = x[..., ::2].float()
    x_odd = x[..., 1::2].float()

    # (T, head_dim/2) -> (1, T, 1, head_dim/2) for broadcasting
    cos = cos[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    sin = sin[: x.shape[1]].unsqueeze(0).unsqueeze(2)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    x_rot = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return x_rot.to(x.dtype)


# GQA
class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads
        self.head_dim   = cfg.dim // cfg.n_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim,    bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
        return_cache: bool = False,
    ):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
        new_cache_k, new_cache_v = (k, v) if return_cache else (None, None)

        # Expand KV heads → Q heads
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # (B, n_heads, T, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash / memory-efficient attention — never materialises (B,H,T,T) score matrix
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=(cache_k is None),   # causal during training; non-causal with cache
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache_k, new_cache_v


# SwiGLU FFN
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.ffn_dim_multiplier)
        hidden = (hidden + 255) & ~255
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn      = GQAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn       = SwiGLU(cfg)

    def _forward(self, x, cos, sin, cache_k, cache_v, return_cache):
        attn_out, nck, ncv = self.attn(
            self.attn_norm(x), cos, sin, cache_k, cache_v, return_cache=return_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, nck, ncv

    def forward(self, x, cos, sin, cache_k=None, cache_v=None, use_grad_ckpt=False, return_cache=False):
        if use_grad_ckpt and self.training:
            # gradient checkpointing: recompute activations on backward instead of storing them
            # cache is None during training so we pass dummy tensors to satisfy checkpoint API
            def ckpt_fn(x, cos, sin):
                out, _, _ = self._forward(x, cos, sin, None, None, False)
                return out
            x = checkpoint(ckpt_fn, x, cos, sin, use_reentrant=False)
            return x, None, None
        return self._forward(x, cos, sin, cache_k, cache_v, return_cache)


# LLM Definition
class LLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight  # weight tying

        head_dim = cfg.dim // cfg.n_heads
        cos, sin = precompute_rope_freqs(head_dim, cfg.max_seq_len * 2, cfg.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[list] = None,
        use_grad_ckpt: bool = False,
        return_cache: bool = False,
    ):
        B, T = idx.shape
        x = self.embed(idx)
        pos_start = 0 if (cache is None or cache[0][0] is None) else cache[0][0].shape[1]
        cos = self.rope_cos[pos_start: pos_start + T]
        sin = self.rope_sin[pos_start: pos_start + T]

        need_cache = return_cache or (cache is not None)
        new_cache = [] if need_cache else None
        for i, layer in enumerate(self.layers):
            ck, cv = cache[i] if cache else (None, None)
            x, nck, ncv = layer(
                x,
                cos,
                sin,
                ck,
                cv,
                use_grad_ckpt=use_grad_ckpt,
                return_cache=need_cache,
            )
            if need_cache:
                new_cache.append((nck, ncv))

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_cache

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def probe_attention_entropy(self, idx: torch.Tensor, max_probe_len: int = 256) -> float:
        """
        Estimate mean causal attention entropy from layer 0 on a short token window.
        Lower entropy means sharper/more concentrated attention.
        """
        if idx.ndim != 2:
            raise ValueError(f"idx must be shape (B, T), got {tuple(idx.shape)}")
        if idx.shape[1] == 0:
            return float("nan")

        probe_len = min(int(max_probe_len), int(idx.shape[1]))
        idx = idx[:, -probe_len:]
        B, T = idx.shape

        x = self.embed(idx)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        layer0 = self.layers[0]
        attn = layer0.attn

        h = layer0.attn_norm(x)
        q = attn.wq(h).view(B, T, attn.n_heads, attn.head_dim)
        k = attn.wk(h).view(B, T, attn.n_kv_heads, attn.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(attn.n_rep, dim=2)

        q = q.transpose(1, 2).float()  # (B, H, T, D)
        k = k.transpose(1, 2).float()  # (B, H, T, D)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        causal_mask = torch.triu(
            torch.ones((T, T), device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        return float(entropy.mean().item())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
    ):
        cache = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:] if cache is None else idx[:, -1:]
            logits, _, cache = self(idx_cond, cache=cache, return_cache=True)
            logits = logits[:, -1, :]

            # Discourage copying previously generated tokens.
            if repetition_penalty > 1.0:
                for b in range(idx.size(0)):
                    used = idx[b].unique()
                    used_logits = logits[b, used]
                    logits[b, used] = torch.where(
                        used_logits > 0, used_logits / repetition_penalty, used_logits * repetition_penalty
                    )

            # Block tokens that would create repeated n-grams.
            if no_repeat_ngram_size and no_repeat_ngram_size > 1 and idx.size(1) >= no_repeat_ngram_size - 1:
                n = int(no_repeat_ngram_size)
                for b in range(idx.size(0)):
                    seq = idx[b].tolist()
                    prefix = tuple(seq[-(n - 1) :])
                    banned = set()
                    for i in range(len(seq) - n + 1):
                        if tuple(seq[i : i + n - 1]) == prefix:
                            banned.add(seq[i + n - 1])
                    if banned:
                        logits[b, list(banned)] = float("-inf")

            if temperature == 0.0:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = sorted_probs.cumsum(-1)
                sorted_probs[cumsum - sorted_probs > top_p] = 0.0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                next_tok = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

            idx = torch.cat([idx, next_tok], dim=1)
        return idx
