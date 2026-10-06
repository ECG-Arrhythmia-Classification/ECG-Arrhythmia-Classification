"""Small Transformer classifier for 180-sample ECG heartbeats."""

import math

import torch
from torch import nn


class EncoderBlock(nn.Module):
    """One self-attention and feed-forward encoder block."""

    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, show_attention=False):
        result, weights = self.attention(
            x, x, x, need_weights=show_attention, average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(result))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return (x, weights) if show_attention else x


class ECGTransformer(nn.Module):
    """Turn short ECG patches into tokens, then classify with a Transformer."""

    def __init__(self, input_length=180, classes=5, dimension=32,
                 heads=4, layers=2, dropout=0.2, patch_size=4):
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("dimension must be divisible by heads")
        if input_length % patch_size != 0:
            raise ValueError("input_length must be divisible by patch_size")

        self.input_length = input_length
        self.patch_size = patch_size
        token_count = input_length // patch_size
        self.embedding = nn.Conv1d(1, dimension, kernel_size=patch_size,
                                   stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.register_buffer(
            "position", self._make_position_encoding(token_count + 1, dimension)
        )
        self.blocks = nn.ModuleList(
            [EncoderBlock(dimension, heads, dropout) for _ in range(layers)]
        )
        self.classifier = nn.Linear(dimension, classes)
        nn.init.normal_(self.class_token, std=0.02)

    @staticmethod
    def _make_position_encoding(length, dimension):*

