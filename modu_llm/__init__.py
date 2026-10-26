"""『모두의 LLM with 파이썬』 실습 패키지.
각 장에서 만든 부품의 검증된 버전이 들어 있습니다."""
from .settings import ASSET_REPO, DRIVE_DIR, END, SFT_REPO, STORY_REPO
from .vocab import Vocab, hangul_2350
from .data import encode_texts, get_batch, load_stories, split_data
from .model_basic import Block, Embeddings, GPTConfig, Head, MultiHead, TinyGPT
from .model import GPT, CausalSelfAttention, FastBlock, copy_basic_to_fast, count_params
from .generate import generate, next_char_probs
from .train import estimate_loss, load_checkpoint, save_checkpoint, train
from .sft import format_pair, load_sft_texts
from .files import get_file, mount_drive
from .viz import show_weights
from .bench import time_steps

__version__ = "0.1.0"

__all__ = [
    "ASSET_REPO", "DRIVE_DIR", "END", "SFT_REPO", "STORY_REPO",
    "Vocab", "hangul_2350", "encode_texts", "get_batch", "load_stories", "split_data",
    "Block", "Embeddings", "GPTConfig", "Head", "MultiHead", "TinyGPT",
    "GPT", "CausalSelfAttention", "FastBlock", "copy_basic_to_fast", "count_params",
    "generate", "next_char_probs",
    "estimate_loss", "load_checkpoint", "save_checkpoint", "train",
    "format_pair", "load_sft_texts", "get_file", "mount_drive", "show_weights", "time_steps",
]
