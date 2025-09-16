#!/usr/bin/env python3
"""Create stratified 7:2:1 train/val/test splits for heart sound datasets."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

RATIOS = OrderedDict([("train", 0.7), ("val", 0.2), ("test", 0.1)])

LABEL_MAP_CIRCOR = {
    "Absent": "absent",
    "Present": "present",
    "Unknown": "unknown",
}

LABEL_MAP_PHYSIONET_2016 = {
    "1": "absent",   # normal heart sound → murmur absent
    "-1": "present",  # abnormal heart sound → murmur present
    "0": "unknown",   # unsure label treated as murmur status unknown
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circor-root",
        type=Path,
        default=Path("/map-vepfs/qinyu/CodeSpace/datasets/physionet-circor-heart-sound/files/circor-heart-sound/1.0.3"),
        help="Path to the CirCor heart sound dataset root (contains training_data.csv).",
    )
    parser.add_argument(
        "--physionet2016-root",
        type=Path,
        default=Path("/map-vepfs/qinyu/CodeSpace/datasets/physionet-challenge-2016/files/challenge-2016/1.0.0"),
        help="Path to the PhysioNet 2016 Challenge dataset root (contains training-a ... training-f).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("metadata"),
        help="Directory where the combined metadata files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed used for deterministic stratified splitting.",
    )
    return parser.parse_args()


def stratified_group_split(group_to_label: Dict[str, str], seed: int) -> Dict[str, str]:
    """Assign each group to train/val/test preserving label ratios."""
    rng = random.Random(seed)
    label_to_groups: Dict[str, List[str]] = defaultdict(list)
    for group, label in group_to_label.items():
        label_to_groups[label].append(group)

    assignments: Dict[str, str] = {}
    for label, groups in label_to_groups.items():
        groups.sort()  # ensure deterministic order before shuffling
        rng.shuffle(groups)
        total = len(groups)
        if total == 0:
            continue
        # compute target counts using largest remainder method
        counts: Dict[str, int] = {}
        remainders: List[Tuple[float, str]] = []
        for split, ratio in RATIOS.items():
            exact = total * ratio
            count = math.floor(exact)
            counts[split] = count
            remainders.append((exact - count, split))
        assigned = sum(counts.values())
        remainder_needed = total - assigned
        # assign extra items to splits with largest fractional remainder
        if remainder_needed > 0:
            remainders.sort(key=lambda x: x[0], reverse=True)
            for _, split in remainders[:remainder_needed]:
                counts[split] += 1
        # assign groups sequentially into splits
        cursor = 0
        for split in RATIOS.keys():
            take = counts.get(split, 0)
            for g in groups[cursor : cursor + take]:
                assignments[g] = split
            cursor += take
    return assignments


def load_circor_entries(root: Path) -> List[Dict[str, object]]:
    metadata_file = root / "training_data.csv"
    audio_root = root / "training_data"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Could not locate CirCor metadata file: {metadata_file}")
    entries: List[Dict[str, object]] = []
    with metadata_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["Patient ID"].strip()
            murmur_label = row["Murmur"].strip().title()
            label = LABEL_MAP_CIRCOR.get(murmur_label)
            if label is None:
                # treat missing annotation as unknown by default
                label = "unknown"
            locations = [
                loc.strip()
                for loc in row["Recording locations:"].split("+")
                if loc.strip()
            ]
            if not locations:
                # fallback to glob search if metadata missing
                locations = [
                    p.stem.split("_")[1]
                    for p in sorted(audio_root.glob(f"{patient_id}_*.wav"))
                    if "_" in p.stem
                ]
            demographics = {
                "age": row.get("Age", ""),
                "sex": row.get("Sex", ""),
                "height": row.get("Height", ""),
                "weight": row.get("Weight", ""),
                "pregnancy_status": row.get("Pregnancy status", ""),
                "outcome": row.get("Outcome", ""),
                "campaign": row.get("Campaign", ""),
            }
            for loc in locations:
                wav_path = audio_root / f"{patient_id}_{loc}.wav"
                if not wav_path.exists():
                    # some patients may miss certain locations; skip quietly
                    continue
                entry = {
                    "dataset": "circor",
                    "patient_id": patient_id,
                    "recording_id": f"{patient_id}_{loc}",
                    "rel_path": str(wav_path.relative_to(root)),
                    "abs_path": str(wav_path.resolve()),
                    "label": label,
                    "location": loc,
                    "demographics": demographics,
                }
                entries.append(entry)
    return entries


def load_physionet2016_entries(root: Path) -> List[Dict[str, object]]:
    subsets = [
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("training-")
    ]
    entries: List[Dict[str, object]] = []
    for subset in sorted(subsets):
        reference_file = subset / "REFERENCE.csv"
        if not reference_file.exists():
            continue
        with reference_file.open("r", newline="") as f:
            reader = csv.reader(f)
            for record_id, raw_label in reader:
                label = LABEL_MAP_PHYSIONET_2016.get(raw_label.strip())
                if label is None:
                    # skip records without a known mapping
                    continue
                wav_path = subset / f"{record_id}.wav"
                if not wav_path.exists():
                    # prefer wav; fall back to wfdb if necessary
                    dat_path = subset / f"{record_id}.dat"
                    if dat_path.exists():
                        wav_path = dat_path
                    else:
                        continue
                entry = {
                    "dataset": "physionet2016",
                    "patient_id": record_id,
                    "recording_id": record_id,
                    "rel_path": str(wav_path.relative_to(root)),
                    "abs_path": str(wav_path.resolve()),
                    "label": label,
                    "location": "unknown",
                    "demographics": {},
                }
                entries.append(entry)
    return entries


def main() -> None:
    args = parse_args()
    entries: List[Dict[str, object]] = []
    entries.extend(load_circor_entries(args.circor_root))
    entries.extend(load_physionet2016_entries(args.physionet2016_root))

    if not entries:
        raise RuntimeError("No entries collected from the provided dataset roots.")

    # assign groups at patient level within each dataset to avoid leakage
    group_to_label: Dict[str, str] = {}
    for entry in entries:
        group_key = f"{entry['dataset']}::{entry['patient_id']}"
        # ensure consistency if patient has multiple recordings
        existing = group_to_label.get(group_key)
        if existing is not None and existing != entry["label"]:
            raise ValueError(
                f"Conflicting labels for patient {group_key}: {existing} vs {entry['label']}"
            )
        group_to_label[group_key] = entry["label"]

    assignments = stratified_group_split(group_to_label, seed=args.seed)
    for entry in entries:
        group_key = f"{entry['dataset']}::{entry['patient_id']}"
        entry["split"] = assignments[group_key]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "combined_metadata.json"
    csv_path = args.out_dir / "combined_metadata.csv"

    json_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    # write CSV header and rows for convenience
    fieldnames = [
        "dataset",
        "split",
        "patient_id",
        "recording_id",
        "label",
        "rel_path",
        "location",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                "dataset": entry["dataset"],
                "split": entry["split"],
                "patient_id": entry["patient_id"],
                "recording_id": entry["recording_id"],
                "label": entry["label"],
                "rel_path": entry["rel_path"],
                "location": entry["location"],
            })

    print(f"Wrote {len(entries)} entries to {json_path} and {csv_path}.")


if __name__ == "__main__":
    main()
