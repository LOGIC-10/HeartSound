"""Dataset utilities for heart sound murmur classification."""
from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torchaudio
from torchaudio import functional as F
from torch.utils.data import Dataset

_LABEL_TO_INDEX = {"absent": 0, "present": 1, "unknown": 2}
_INDEX_TO_LABEL = {v: k for k, v in _LABEL_TO_INDEX.items()}


@dataclass
class Segment:
    path: Path
    location: str


@dataclass
class PatientRecord:
    dataset: str
    patient_id: str
    label: str
    segments: List[Segment]
    demographics: Dict[str, str]

    @property
    def label_index(self) -> int:
        try:
            return _LABEL_TO_INDEX[self.label]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown label: {self.label}") from exc


class HeartSoundDataset(Dataset):
    """Patient-level dataset that aggregates multi-location heart sound recordings.

    Enhancements (2025 recipe):
    - Train-time audio augmentations (random gain, circular time shift, band-reject, Gaussian noise with SNR control)
    - Train-time SpecAugment on Mel features (time/frequency masking)
    """

    def __init__(
        self,
        metadata_path: Path,
        split: str,
        sample_rate: int = 4000,
        max_duration: float = 20.0,
        max_locations: int = 4,
        device: torch.device | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * max_duration)
        self.max_locations = max_locations
        self.device = device
        self.is_train = split == "train"

        raw_entries = json.loads(self.metadata_path.read_text())
        grouped: Dict[str, PatientRecord] = {}
        for entry in raw_entries:
            if entry.get("split") != split:
                continue
            key = f"{entry['dataset']}::{entry['patient_id']}"
            segment = Segment(path=Path(entry["abs_path"]), location=entry.get("location", ""))
            if key not in grouped:
                grouped[key] = PatientRecord(
                    dataset=entry["dataset"],
                    patient_id=entry["patient_id"],
                    label=entry["label"],
                    segments=[segment],
                    demographics=entry.get("demographics", {}),
                )
            else:
                grouped[key].segments.append(segment)
        self.records: List[PatientRecord] = sorted(grouped.values(), key=lambda r: (r.dataset, r.patient_id))

        # Pre-compute reusable transforms
        self.mel_transforms = [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=n_mels,
                f_min=20.0,
                f_max=1600.0,
                power=2.0,
            )
            for n_mels in (64, 128)
        ]
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # SpecAugment (applied only during training on Mel)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)

    def __len__(self) -> int:
        return len(self.records)

    def _random_circular_shift(self, waveform: torch.Tensor, max_frac: float = 0.1) -> torch.Tensor:
        if not self.is_train or max_frac <= 0:
            return waveform
        max_shift = int(self.target_samples * max_frac)
        if max_shift <= 0:
            return waveform
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if shift == 0:
            return waveform
        return torch.roll(waveform, shifts=shift, dims=-1)

    def _random_gain(self, waveform: torch.Tensor, low: float = 0.6, high: float = 1.4) -> torch.Tensor:
        if not self.is_train:
            return waveform
        gain = float(torch.empty(1).uniform_(low, high).item())
        return (waveform * gain).clamp_(-1.0, 1.0)

    def _random_bandreject(self, waveform: torch.Tensor, p: float = 0.3) -> torch.Tensor:
        if not self.is_train or torch.rand(1).item() > p:
            return waveform
        # pick a random stop band within 50-800 Hz typical PCG bandwidth
        f1 = float(torch.empty(1).uniform_(50.0, 400.0).item())
        f2 = float(torch.empty(1).uniform_(f1 + 50.0, 800.0).item())
        # approximate band-reject via sequential high/low pass
        w = torchaudio.functional.highpass_biquad(waveform, self.sample_rate, cutoff_freq=f2)
        w = torchaudio.functional.lowpass_biquad(w, self.sample_rate, cutoff_freq=f1)
        # mix residual to avoid over-suppression
        alpha = 0.5
        return (alpha * w + (1 - alpha) * waveform).clamp_(-1.0, 1.0)

    def _add_gaussian_noise(self, waveform: torch.Tensor, snr_db_range=(15.0, 30.0), p: float = 0.7) -> torch.Tensor:
        if not self.is_train or torch.rand(1).item() > p:
            return waveform
        snr_db = float(torch.empty(1).uniform_(snr_db_range[0], snr_db_range[1]).item())
        sig_power = (waveform ** 2).mean().clamp(min=1e-8)
        snr = 10 ** (snr_db / 10.0)
        noise_power = (sig_power / snr).item()
        noise = torch.randn_like(waveform) * math.sqrt(noise_power)
        return (waveform + noise).clamp_(-1.0, 1.0)

    def _load_waveform(self, path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        waveform = F.highpass_biquad(waveform, self.sample_rate, cutoff_freq=20.0)
        waveform = F.lowpass_biquad(waveform, self.sample_rate, cutoff_freq=800.0)
        waveform = waveform / (waveform.abs().max() + 1e-6)
        # Train-time augmentations
        waveform = self._random_circular_shift(waveform)
        waveform = self._random_gain(waveform)
        waveform = self._random_bandreject(waveform)
        waveform = self._add_gaussian_noise(waveform)
        return waveform

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[-1]
        if length > self.target_samples:
            waveform = waveform[..., : self.target_samples]
        elif length < self.target_samples:
            pad_length = self.target_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def _compute_mel_features(self, waveform: torch.Tensor) -> torch.Tensor:
        features = []
        for transform in self.mel_transforms:
            mel = transform(waveform)
            mel_db = self.amplitude_to_db(mel + 1e-10)
            if self.is_train:
                # SpecAugment in-place on each mel band
                mel_db = self.freq_mask(mel_db)
                mel_db = self.time_mask(mel_db)
            features.append(mel_db)
        return torch.cat(features, dim=1)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | Dict[str, str]]:
        record = self.records[index]
        waveforms: List[torch.Tensor] = []
        mel_features: List[torch.Tensor] = []
        for segment in record.segments[: self.max_locations]:
            waveform = self._pad_or_trim(self._load_waveform(segment.path))
            mel = self._compute_mel_features(waveform)
            waveforms.append(waveform)
            mel_features.append(mel)
        num_segments = len(waveforms)
        if num_segments == 0:
            raise RuntimeError(f"No audio segments found for patient {record.patient_id}")
        waveform_tensor = torch.zeros(self.max_locations, 1, self.target_samples)
        mel_tensor = torch.zeros(self.max_locations, mel_features[0].shape[1], mel_features[0].shape[2])
        mask = torch.zeros(self.max_locations, dtype=torch.bool)
        for i in range(num_segments):
            waveform_tensor[i] = waveforms[i]
            mel_tensor[i] = mel_features[i]
            mask[i] = True
        label_index = record.label_index
        sample = {
            "waveforms": waveform_tensor.to(self.device) if self.device else waveform_tensor,
            "mel": mel_tensor.to(self.device) if self.device else mel_tensor,
            "mask": mask.to(self.device) if self.device else mask,
            "label": torch.tensor(label_index, dtype=torch.long, device=self.device) if self.device else torch.tensor(label_index, dtype=torch.long),
            "metadata": {
                "dataset": record.dataset,
                "patient_id": record.patient_id,
                **{k: v for k, v in record.demographics.items() if v},
            },
        }
        return sample


def collate_fn(batch: Sequence[Dict[str, torch.Tensor | Dict[str, str]]]) -> Dict[str, torch.Tensor | List[Dict[str, str]]]:
    waveforms = torch.stack([item["waveforms"] for item in batch])
    mel = torch.stack([item["mel"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    metadata = [item["metadata"] for item in batch]
    return {"waveforms": waveforms, "mel": mel, "mask": mask, "labels": labels, "metadata": metadata}


__all__ = [
    "HeartSoundDataset",
    "collate_fn",
    "_LABEL_TO_INDEX",
    "_INDEX_TO_LABEL",
]
