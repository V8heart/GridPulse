"""Small decoder-only GPT (SDPA) presets for local capture — no downloads."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GPTPreset:
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int = 1024
    block_size: int = 128


PRESETS: dict[str, GPTPreset] = {
    "tiny": GPTPreset("tiny", n_layer=2, n_head=2, n_embd=64, vocab_size=512, block_size=64),
    # Capture default. Wider than the old toy small so training power is not
    # trivially separable from SWMA/crypto matmuls. medium stays unused.
    "small": GPTPreset("small", n_layer=8, n_head=12, n_embd=768, vocab_size=1024, block_size=128),
    "medium": GPTPreset("medium", n_layer=6, n_head=8, n_embd=256, vocab_size=2048, block_size=256),
}


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        past_len = 0
        if past is not None:
            past_len = past[0].size(-2)
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        # SDPA when available; fallback to manual masked attention.
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            if past_len == 0:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=0.0
                )
            else:
                keys = torch.arange(k.size(-2), device=x.device)
                queries = past_len + torch.arange(t, device=x.device)
                causal_mask = keys.unsqueeze(0) <= queries.unsqueeze(1)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=causal_mask, dropout_p=0.0
                )
        else:
            att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            keys = torch.arange(k.size(-2), device=x.device)
            queries = past_len + torch.arange(t, device=x.device)
            causal_mask = keys.unsqueeze(0) <= queries.unsqueeze(1)
            att = att.masked_fill(~causal_mask, float("-inf"))
            att = torch.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        output = self.proj(y)
        return (output, (k, v)) if use_cache else output


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended = self.attn(self.ln1(x), past, use_cache=use_cache)
        if use_cache:
            attended, present = attended
        x = x + attended
        x = x + self.mlp(self.ln2(x))
        return (x, present) if use_cache else x


class TinyGPT(nn.Module):
    def __init__(self, preset: GPTPreset | str = "tiny"):
        super().__init__()
        cfg = PRESETS[preset] if isinstance(preset, str) else preset
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            [Block(cfg.n_embd, cfg.n_head, cfg.block_size) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(
        self,
        idx: torch.Tensor,
        *,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        _b, t = idx.shape
        past_len = 0 if not past_key_values else past_key_values[0][0].size(-2)
        if past_len + t > self.cfg.block_size:
            raise ValueError(f"sequence length {past_len + t} > block_size {self.cfg.block_size}")
        pos = torch.arange(past_len, past_len + t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            result = block(x, past, use_cache=use_cache)
            if use_cache:
                x, present = result
                presents.append(present)
            else:
                x = result
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return (logits, presents) if use_cache else logits


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_gpt(preset: str = "tiny", *, device: str | torch.device = "cpu") -> TinyGPT:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    model = TinyGPT(preset)
    return model.to(device)
