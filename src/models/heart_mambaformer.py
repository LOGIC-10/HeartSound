"""Heart-MambaFormer: hybrid backbone for heart sound murmur classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

from src.data.dataset import _INDEX_TO_LABEL


@dataclass
class HeartMambaConfig:
    hidden_dim: int = 256
    state_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    waveform_backbone: Optional[str] = "microsoft/wavlm-base-plus"
    backbone_sample_rate: int = 16000
    freeze_backbone: bool = True
    metadata_dim: int = 64
    # Phase/Cycle aware options
    phase_gating: bool = False
    phase_kernel: int = 15
    phase_strength: float = 0.5  # 0..1


class SimpleSSMBlock(nn.Module):
    """Lightweight state-space inspired block using cumulative filtering."""

    def __init__(self, dim: int, state_dim: int, dropout: float) -> None:
        super().__init__()
        self.value_proj = nn.Linear(dim, state_dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(state_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.register_parameter("decay", nn.Parameter(torch.zeros(state_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, dim)
        residual = x
        gate = torch.sigmoid(self.gate_proj(x))
        values = torch.tanh(self.value_proj(x))
        decay = torch.sigmoid(self.decay).view(1, 1, -1)
        # Exponential moving average implemented via cumulative product
        cumulative = torch.cumsum(values * (1 - decay), dim=1)
        filtered = self.out_proj(cumulative)
        filtered = self.dropout(gate * filtered)
        return self.norm(residual + filtered)


class MetadataEncoder(nn.Module):
    """Encodes demographic metadata into dense feature vectors."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.dataset_vocab = {"circor": 0, "physionet2016": 1, "unknown": 2}
        self.sex_vocab = {"Female": 0, "Male": 1, "Unknown": 2, "": 2}
        self.age_vocab = {"Neonate": 0, "Infant": 1, "Child": 2, "Adolescent": 3, "Adult": 4, "": 5, "Unknown": 5}
        self.pregnancy_vocab = {"True": 1, "False": 0, "": 2, "Unknown": 2}
        embed_dim = output_dim // 4
        self.dataset_embed = nn.Embedding(len(self.dataset_vocab), embed_dim)
        self.sex_embed = nn.Embedding(len(self.sex_vocab), embed_dim)
        self.age_embed = nn.Embedding(len(self.age_vocab), embed_dim)
        self.pregnancy_embed = nn.Embedding(len(self.pregnancy_vocab), embed_dim)
        self.proj = nn.Linear(embed_dim * 4 + 3, output_dim)

    def _lookup(self, vocab: Dict[str, int], key: str) -> torch.Tensor:
        index = vocab.get(key, vocab.get("Unknown", vocab.get("", 0)))
        return torch.tensor(index, dtype=torch.long)

    def forward(self, metadata: List[Dict[str, str]], device: torch.device) -> torch.Tensor:
        embeddings = []
        for meta in metadata:
            dataset_embed = self.dataset_embed(self._lookup(self.dataset_vocab, meta.get("dataset", "unknown")).to(device))
            sex_embed = self.sex_embed(self._lookup(self.sex_vocab, meta.get("sex", "")).to(device))
            age_embed = self.age_embed(self._lookup(self.age_vocab, meta.get("age", "")).to(device))
            pregnancy_embed = self.pregnancy_embed(self._lookup(self.pregnancy_vocab, meta.get("pregnancy_status", "")).to(device))
            features = [dataset_embed, sex_embed, age_embed, pregnancy_embed]
            numeric = []
            for key in ("height", "weight"):
                value = meta.get(key, "")
                try:
                    numeric.append(float(value))
                except (TypeError, ValueError):
                    numeric.append(0.0)
            # encode patient age if numeric value available
            age_numeric = 0.0
            try:
                age_numeric = float(meta.get("age_numeric", 0.0))
            except (TypeError, ValueError):
                age_numeric = 0.0
            numeric.append(age_numeric)
            numeric_tensor = torch.tensor(numeric, dtype=torch.float32, device=device)
            cat_embed = torch.cat(features)
            embeddings.append(torch.cat([cat_embed, numeric_tensor]))
        stacked = torch.stack(embeddings).to(device)
        return F.silu(self.proj(stacked))


class LocationAttentionPool(nn.Module):
    """Cross-location attention pooling with learnable queries."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: (batch, locations, hidden)
        query = self.query.expand(features.size(0), -1, -1)
        pooled, _ = self.attn(query, features, features, key_padding_mask=~mask)
        return pooled.squeeze(1)


class HeartMambaFormer(nn.Module):
    def __init__(self, config: HeartMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels=192, out_channels=config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.ssm_layers = nn.ModuleList(
            [SimpleSSMBlock(config.hidden_dim, config.state_dim, config.dropout) for _ in range(config.depth)]
        )
        self.location_pool = LocationAttentionPool(config.hidden_dim, config.num_heads)
        self.metadata_encoder = MetadataEncoder(config.metadata_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim + config.metadata_dim),
            nn.Linear(config.hidden_dim + config.metadata_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(_INDEX_TO_LABEL)),
        )

        self.waveform_backbone = None
        self.waveform_resampler = None
        if config.waveform_backbone:
            self.waveform_backbone = AutoModel.from_pretrained(config.waveform_backbone)
            if config.freeze_backbone:
                for param in self.waveform_backbone.parameters():
                    param.requires_grad = False
            self.waveform_resampler = torchaudio.transforms.Resample(orig_freq=4000, new_freq=config.backbone_sample_rate)
            self.backbone_proj = nn.Linear(self.waveform_backbone.config.hidden_size, config.hidden_dim)

        # phase gating smoothing kernel (on Mel frame axis)
        self.phase_kernel = max(1, int(config.phase_kernel))
        if self.phase_kernel % 2 == 0:
            self.phase_kernel += 1  # ensure odd for symmetric smoothing

    def _encode_waveform_backbone(self, waveforms: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.waveform_backbone is None:
            return torch.zeros(waveforms.size(0), waveforms.size(1), self.config.hidden_dim, device=waveforms.device)
        batch, locations, _ = waveforms.shape
        flattened = waveforms.view(batch * locations, -1)
        if self.waveform_resampler is not None:
            resampled = self.waveform_resampler.to(waveforms.device)(flattened)
        else:
            resampled = flattened
        outputs = self.waveform_backbone(input_values=resampled)
        hidden = outputs.last_hidden_state.mean(dim=1)
        hidden = self.backbone_proj(hidden)
        hidden = hidden.view(batch, locations, -1)
        hidden = hidden * mask.unsqueeze(-1)
        return hidden

    def _phase_weights_from_waveform(self, waveforms: torch.Tensor, t_frames: int, device: torch.device) -> torch.Tensor:
        """Compute normalized energy envelope per location and downsample to Mel frames.

        Args:
            waveforms: (B, L, S) at 4kHz
            t_frames: target number of Mel frames
        Returns:
            weights: (B*L, t_frames, 1) in [0,1], smoothed
        """
        b, l, s = waveforms.shape
        x = waveforms.view(b * l, 1, s).abs()  # (B*L, 1, S)
        # Downsample by approximate hop length used in Mel transform (256 at 4kHz)
        hop = max(1, s // max(1, t_frames))
        k = hop
        env = torch.nn.functional.avg_pool1d(x, kernel_size=k, stride=hop, padding=k // 2)
        # Adjust length to exactly t_frames
        if env.shape[-1] != t_frames:
            env = torch.nn.functional.interpolate(env, size=t_frames, mode="linear", align_corners=False)
        # Normalize per sample to [0,1]
        env = env.squeeze(1)  # (B*L, T)
        min_v = env.amin(dim=-1, keepdim=True)
        max_v = env.amax(dim=-1, keepdim=True)
        weights = (env - min_v) / (max_v - min_v + 1e-6)
        # Smooth on frame axis
        pad = self.phase_kernel // 2
        weights = torch.nn.functional.pad(weights.unsqueeze(1), (pad, pad), mode="reflect")
        kernel = torch.ones(1, 1, self.phase_kernel, device=device) / float(self.phase_kernel)
        weights = torch.nn.functional.conv1d(weights, kernel).squeeze(1)
        # Clip to [0,1]
        weights = weights.clamp(0.0, 1.0)
        return weights.unsqueeze(-1)  # (B*L, T, 1)

    def forward(self, batch: Dict[str, torch.Tensor | List[Dict[str, str]]]) -> Dict[str, torch.Tensor]:
        mel: torch.Tensor = batch["mel"]  # (B, L, C, T)
        mask: torch.Tensor = batch["mask"]  # (B, L)
        labels: torch.Tensor = batch["labels"]
        metadata: List[Dict[str, str]] = batch["metadata"]  # type: ignore[assignment]
        waveforms_full: torch.Tensor = batch["waveforms"]  # (B, L, 1, S)

        batch_size, num_locations, _, time_dim = mel.shape
        mel = mel.view(batch_size * num_locations, -1, time_dim)
        features = self.input_proj(mel)
        features = features.transpose(1, 2)  # (B*L, T, hidden)
        # Phase-aware gating on temporal features using waveform energy envelope
        if self.config.phase_gating:
            with torch.no_grad():
                # detach waveforms for gating weights
                wave_w = waveforms_full.squeeze(2).to(mel.device)
                weights = self._phase_weights_from_waveform(wave_w, features.size(1), device=mel.device)
            strength = float(self.config.phase_strength)
            features = features * (1.0 - strength + strength * weights)
        for layer in self.ssm_layers:
            features = layer(features)
        temporal_pooled = features.mean(dim=1)
        temporal_pooled = temporal_pooled.view(batch_size, num_locations, -1)
        temporal_pooled = temporal_pooled * mask.unsqueeze(-1)

        waveform_features = self._encode_waveform_backbone(waveforms_full.squeeze(2), mask)
        fused = temporal_pooled + waveform_features

        pooled = self.location_pool(fused, mask)
        device = pooled.device
        metadata_emb = self.metadata_encoder(metadata, device)
        logits = self.classifier(torch.cat([pooled, metadata_emb], dim=-1))
        return {
            "logits": logits,
            "labels": labels,
            "pooled_features": pooled,
        }


__all__ = ["HeartMambaFormer", "HeartMambaConfig"]
