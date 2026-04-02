from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.tanh(self.proj(x))
        attention = self.score(attention).squeeze(-1)
        weights = torch.softmax(attention, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.se = SqueezeExcite1D(out_channels)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        return F.gelu(x + residual)


class RawWindowEncoder(nn.Module):
    def __init__(self, num_channels: int = 18, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.backbone = nn.Sequential(
            ResidualTemporalBlock(32, 64, kernel_size=9, stride=2, dropout=0.10),
            ResidualTemporalBlock(64, 96, kernel_size=7, stride=2, dropout=0.15),
            ResidualTemporalBlock(96, emb_dim, kernel_size=5, stride=2, dropout=0.20),
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.attn = TemporalAttention(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return self.attn(x)


class RawWindowCNNSeqBiGRU(nn.Module):
    def __init__(
        self,
        num_channels: int = 18,
        seq_len: int = 8,
        window_emb_dim: int = 128,
        seq_hidden_dim: int = 128,
        num_classes: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.window_encoder = RawWindowEncoder(num_channels=num_channels, emb_dim=window_emb_dim)
        self.pre_gru_norm = nn.LayerNorm(window_emb_dim)
        self.gru = nn.GRU(
            input_size=window_emb_dim,
            hidden_size=seq_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        self.post_gru_norm = nn.LayerNorm(seq_hidden_dim * 2)
        self.seq_attn = TemporalAttention(seq_hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(seq_hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, samples = x.shape
        x = x.reshape(batch_size * steps, channels, samples)
        x = self.window_encoder(x)
        x = x.reshape(batch_size, steps, -1)
        x = self.pre_gru_norm(x)
        x, _ = self.gru(x)
        x = self.post_gru_norm(x)
        x = self.seq_attn(x)
        return self.classifier(x)


@dataclass
class ModelLoadResult:
    model: nn.Module
    checkpoint_metadata: dict[str, Any]


def build_model(
    model_version: str,
    num_channels: int,
    seq_len: int,
) -> nn.Module:
    if model_version != "raw-window-cnn-seqbigru-v1":
        raise ValueError(
            f"Unsupported model version '{model_version}'. "
            "This app currently supports the raw-window-cnn-seqbigru-v1 checkpoint format."
        )
    return RawWindowCNNSeqBiGRU(num_channels=num_channels, seq_len=seq_len)


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
            return checkpoint["state_dict"], metadata
        if "model_state_dict" in checkpoint:
            metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
            return checkpoint["model_state_dict"], metadata
        tensor_values = [value for value in checkpoint.values() if isinstance(value, torch.Tensor)]
        if tensor_values:
            return checkpoint, {}

    raise ValueError(
        "Checkpoint format not recognized. Expected a state_dict or a dict containing "
        "'state_dict' or 'model_state_dict'."
    )


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict
