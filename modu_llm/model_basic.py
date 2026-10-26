"""3~7장: 이해하기 쉬운 기본 GPT. (학습에는 8장의 빠른 GPT를 씁니다.)
부품은 문장 하나(글자, 숫자)와 배치(문장, 글자, 숫자)를 모두 받습니다."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2453   # Vocab.build()의 글자 수
    block_size: int = 384    # 한 번에 보는 글자 수 (문맥 길이)
    n_embd: int = 256        # 글자 하나를 나타내는 벡터의 길이
    n_head: int = 4          # 어텐션 헤드 수
    n_layer: int = 6         # Transformer 블록 수


class Embeddings(nn.Module):
    """3장: 글자 임베딩 + 위치 임베딩."""

    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

    def forward(self, idx):
        T = idx.shape[-1]
        pos = torch.arange(T, device=idx.device)
        tok = self.token_emb(idx)
        p = self.pos_emb(pos)
        return tok + p


class Head(nn.Module):
    """4장: 어텐션 헤드 하나."""

    def __init__(self, cfg, head_size):
        super().__init__()
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)

    def forward(self, x):
        T = x.shape[-2]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)                  # ① 관련도
        scores = scores / k.shape[-1] ** 0.5
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))  # ② 미래 가리기
        weights = F.softmax(scores, dim=-1)               # ③ 비율로 바꾸기
        self.weights = weights                            # 비율 표를 남겨 두기
        out = weights @ v                                 # ④ 가중합
        return out


class MultiHead(nn.Module):
    """5장: 헤드 여러 개를 나란히."""

    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList()
        for i in range(cfg.n_head):
            self.heads.append(Head(cfg, head_size))
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        outs = []
        for h in self.heads:
            outs.append(h(x))
        out = torch.cat(outs, dim=-1)
        out = self.proj(out)
        return out


def make_mlp(cfg):
    return nn.Sequential(
        nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
        nn.GELU(),
        nn.Linear(4 * cfg.n_embd, cfg.n_embd),
    )


class Block(nn.Module):
    """6장: Transformer 블록."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHead(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = make_mlp(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """7장: 기본 GPT."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = Embeddings(cfg)
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layer):
            self.blocks.append(Block(cfg))
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.head(x)

        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss
