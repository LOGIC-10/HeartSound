#!/usr/bin/env python3
"""Training script for the Heart-MambaFormer murmur classifier."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import HeartSoundDataset, collate_fn, _LABEL_TO_INDEX
from src.models.heart_mambaformer import HeartMambaConfig, HeartMambaFormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/heart_mambaformer.yaml"))
    parser.add_argument("--output", type=Path, default=None, help="Optional override for logging directory.")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.mean()
    return float(ece)


def weighted_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weight_map = {label: count / total for label, count in zip(values, counts)}
    acc = 0.0
    for label, weight in weight_map.items():
        mask = labels == label
        if mask.sum() == 0:
            continue
        acc += weight * (preds[mask] == labels[mask]).mean()
    return float(acc)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "weighted_accuracy": weighted_accuracy(labels, preds),
        "ece": expected_calibration_error(probs, labels),
    }
    try:
        metrics["auroc_macro"] = roc_auc_score(labels, probs, multi_class="ovr")
    except ValueError:
        metrics["auroc_macro"] = float("nan")
    try:
        metrics["auprc_macro"] = average_precision_score(labels, probs, average="macro")
    except ValueError:
        metrics["auprc_macro"] = float("nan")
    return metrics


def prepare_dataloaders(config: Dict, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_kwargs = dict(
        metadata_path=Path(config["metadata_path"]),
        sample_rate=config["sample_rate"],
        max_duration=config["max_duration"],
        max_locations=config["max_locations"],
    )
    train_dataset = HeartSoundDataset(split="train", **dataset_kwargs)
    val_dataset = HeartSoundDataset(split="val", **dataset_kwargs)
    test_dataset = HeartSoundDataset(split="test", **dataset_kwargs)

    common_loader_kwargs = dict(
        num_workers=accelerator.state.num_processes,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=False, **common_loader_kwargs)
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    train_loader: DataLoader,
    accelerator: Accelerator,
    config: Dict,
) -> float:
    model.train()
    total_loss = 0.0
    grad_accum = config["train"].get("gradient_accumulation_steps", 1)
    contrastive_weight = config["train"].get("contrastive_weight", 0.0)
    temperature = config["train"].get("contrastive_temperature", 0.3)
    label_smoothing = config["train"].get("label_smoothing", 0.0)
    loss_type = config["train"].get("loss_type", "ce")

    # Optional: class-balanced focal loss for long-tail robustness
    def focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        probs = F.softmax(logits, dim=-1)
        targets_onehot = F.one_hot(targets, num_classes=probs.size(-1)).float()
        pt = (probs * targets_onehot).sum(dim=-1).clamp_min(1e-8)
        focal_factor = (1.0 - pt) ** gamma
        ce = F.cross_entropy(logits, targets, reduction="none")
        if alpha is not None:
            alpha_factor = alpha[targets]
            ce = alpha_factor * ce
        return (focal_factor * ce).mean()
    for step, batch in enumerate(train_loader):
        batch = move_batch_to_device(batch, accelerator.device)
        outputs = model(batch)
        logits = outputs["logits"]
        labels = outputs["labels"]
        if loss_type == "focal":
            gamma = float(config["train"].get("focal_gamma", 2.0))
            alpha_tensor: Optional[torch.Tensor] = None
            if config["train"].get("focal_alpha_balanced", False):
                # Build alpha weights from label priors in the current batch as fallback.
                # If global priors are desired, compute offline and pass via config.
                num_classes = logits.size(-1)
                counts = torch.bincount(labels, minlength=num_classes).float()
                priors = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-8)
                alpha = (1.0 / priors)
                alpha = alpha / alpha.mean().clamp_min(1e-8)
                alpha_tensor = alpha.to(logits.device)
            ce_loss = focal_loss(logits, labels, gamma=gamma, alpha=alpha_tensor)
        else:
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        loss = ce_loss
        if contrastive_weight > 0:
            features = F.normalize(outputs["pooled_features"], dim=-1)
            logits_sim = torch.matmul(features, features.T) / temperature
            labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
            logits_sim = logits_sim.masked_fill(mask, float("-inf"))
            positives = torch.where(labels_matrix & ~mask, torch.exp(logits_sim), torch.zeros_like(logits_sim))
            negatives = torch.exp(logits_sim)
            denom = negatives.sum(dim=-1) + 1e-8
            numer = positives.sum(dim=-1) + 1e-8
            contrastive_loss = -torch.log(numer / denom).mean()
            loss = loss + contrastive_weight * contrastive_loss
        loss = loss / grad_accum
        accelerator.backward(loss)
        if (step + 1) % grad_accum == 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / max(1, len(train_loader))


def evaluate(model: torch.nn.Module, dataloader: DataLoader, accelerator: Accelerator) -> Dict[str, float]:
    model.eval()
    all_probs: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, accelerator.device)
            outputs = model(batch)
            logits = outputs["logits"]
            labels = outputs["labels"]
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            all_probs.append(accelerator.gather_for_metrics(probs))
            all_preds.append(accelerator.gather_for_metrics(preds))
            all_labels.append(accelerator.gather_for_metrics(labels))
    probs_np = torch.cat(all_probs).cpu().numpy()
    preds_np = torch.cat(all_preds).cpu().numpy()
    labels_np = torch.cat(all_labels).cpu().numpy()
    return compute_metrics(labels_np, probs_np, preds_np)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    accelerator = Accelerator(mixed_precision=config["train"].get("mixed_precision", "no"))
    if args.output:
        config["logging"]["output_dir"] = str(args.output)

    set_seed(config.get("seed", 2025))
    train_loader, val_loader, test_loader = prepare_dataloaders(config, accelerator)

    model_config = HeartMambaConfig(**config["model"])
    model = HeartMambaFormer(model_config)
    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])

    # Warmup + cosine decay scheduler at step granularity
    steps_per_epoch = math.ceil(len(train_loader) / max(1, config["train"].get("gradient_accumulation_steps", 1)))
    total_steps = steps_per_epoch * config["train"]["num_epochs"]
    warmup_epochs = int(config["train"].get("warmup_epochs", 0))
    warmup_steps = warmup_epochs * steps_per_epoch
    min_lr = float(config["scheduler"].get("min_lr", 1e-6))
    base_lr = float(config["train"]["learning_rate"]) 

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(1e-4, (current_step + 1) / float(max(1, warmup_steps)))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = min_lr / max(1e-12, base_lr)
        return floor + (1.0 - floor) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    # test_loader not passed to prepare to keep CPU-based iteration for gather

    output_dir = Path(config["logging"]["output_dir"]) / config["experiment_name"]
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    best_metric = -float("inf")
    best_checkpoint = None
    history: List[Dict[str, float]] = []

    unfreeze_epoch = int(config["train"].get("unfreeze_backbone_at_epoch", 0))
    for epoch in range(1, config["train"]["num_epochs"] + 1):
        accelerator.print(f"Epoch {epoch}/{config['train']['num_epochs']}")
        # Optionally unfreeze backbone for fine-tuning after warmup period
        if (
            unfreeze_epoch > 0
            and epoch == unfreeze_epoch
            and hasattr(model, "waveform_backbone")
            and model.waveform_backbone is not None
        ):
            if accelerator.is_main_process:
                accelerator.print(f"Unfreezing backbone parameters at epoch {epoch}.")
            for p in model.waveform_backbone.parameters():
                p.requires_grad = True
        train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, accelerator, config)
        val_metrics = evaluate(model, val_loader, accelerator)
        val_score = val_metrics["macro_f1"]
        if accelerator.is_main_process:
            metrics_record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            history.append(metrics_record)
            history_path = output_dir / "history.json"
            history_path.write_text(json.dumps(history, indent=2))
        accelerator.print(f"Epoch {epoch} - loss {train_loss:.4f} - val macro-F1 {val_score:.4f}")
        if val_score > best_metric and accelerator.is_main_process:
            best_metric = val_score
            best_checkpoint = output_dir / "best.pt"
            accelerator.save_state(output_dir)
            torch.save(accelerator.unwrap_model(model).state_dict(), best_checkpoint)

    accelerator.wait_for_everyone()
    if best_checkpoint is not None and accelerator.is_main_process:
        accelerator.print(f"Best checkpoint saved at {best_checkpoint}")

    # Final evaluation on test split (run on unprepared loader)
    model.eval()
    test_loader = accelerator.prepare_data_loader(test_loader)
    test_metrics = evaluate(model, test_loader, accelerator)
    if accelerator.is_main_process:
        metrics_path = output_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(test_metrics, indent=2))
        accelerator.print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
