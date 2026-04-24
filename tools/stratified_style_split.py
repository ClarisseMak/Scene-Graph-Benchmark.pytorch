#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Stratified split of images by style_label from fashion_style_mapping.json.

Pipeline:
  1) Filter style_label >= 0
  2) Stratified 70% train / 30% temp (random_state fixed)
  3) From temp: random val_size (default 5000) -> val, remainder -> test

Training: set cfg DATASETS.STYLE_STRATIFIED_SPLIT_FILE to the output JSON path so VGDataset
uses these lists. With DATASETS.STYLE_STRATIFIED_SPLIT_STRICT True (default), train/val/test
follow the JSON image_id lists on full HDF5 rows (HDF5 split flags are ignored). Set STRICT
False for the legacy path: H5 split==0 for train&val and split==2 for test, then intersect
with the JSON lists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from maskrcnn_benchmark.data.datasets.style_tags import QWEN_STYLE_TAGS_V1  # noqa: E402


def _try_sklearn_stratified_split(
    indices: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    return train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )


def _numpy_stratified_split(
    indices: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate stratified split per class (same idea as sklearn)."""
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for c in np.unique(y):
        cls_idx = indices[y == c]
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = int(round(n * test_size))
        n_test = max(0, min(n_test, n))
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def load_labeled_pairs(mapping_path: str) -> Tuple[List[str], np.ndarray]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by = data.get("by_image_id")
    if not isinstance(by, dict):
        raise ValueError("mapping JSON must contain by_image_id dict")

    ids: List[str] = []
    labels: List[int] = []
    for k, v in by.items():
        iid = str(k)
        lab = v.get("style_label", -1) if isinstance(v, dict) else v
        try:
            lab_i = int(lab)
        except (TypeError, ValueError):
            continue
        if lab_i < 0:
            continue
        if lab_i >= len(QWEN_STYLE_TAGS_V1):
            continue
        ids.append(iid)
        labels.append(lab_i)

    if not ids:
        raise ValueError("No valid labeled images (style_label in [0, num_classes-1])")
    return ids, np.asarray(labels, dtype=np.int64)


def split_train_temp_val_test(
    ids: List[str],
    y: np.ndarray,
    train_frac: float,
    val_size: int,
    random_state: int,
) -> Tuple[List[str], List[str], List[str]]:
    n = len(ids)
    indices = np.arange(n, dtype=np.int64)
    test_size = 1.0 - train_frac
    try:
        tr_i, te_i = _try_sklearn_stratified_split(indices, y, test_size=test_size, random_state=random_state)
    except Exception:
        tr_i, te_i = _numpy_stratified_split(indices, y, test_size=test_size, random_state=random_state)

    train_ids = [ids[i] for i in tr_i.tolist()]
    temp_ids = [ids[i] for i in te_i.tolist()]
    y_temp = y[te_i]

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(temp_ids))
    v_n = min(int(val_size), len(temp_ids))
    val_sel = perm[:v_n]
    test_sel = perm[v_n:]
    val_ids = [temp_ids[i] for i in val_sel.tolist()]
    test_ids = [temp_ids[i] for i in test_sel.tolist()]

    return train_ids, val_ids, test_ids


def _counts_for_split(split_ids: List[str], id_to_label: Dict[str, int], num_classes: int) -> List[int]:
    c = [0] * num_classes
    for iid in split_ids:
        lab = id_to_label.get(str(iid), -1)
        if 0 <= lab < num_classes:
            c[lab] += 1
    return c


def _markdown_table(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    id_to_label: Dict[str, int],
) -> str:
    num_classes = len(QWEN_STYLE_TAGS_V1)
    ct = _counts_for_split(train_ids, id_to_label, num_classes)
    cv = _counts_for_split(val_ids, id_to_label, num_classes)
    cs = _counts_for_split(test_ids, id_to_label, num_classes)
    nt, nv, ns = len(train_ids), len(val_ids), len(test_ids)

    rows: List[List[str]] = [
        [
            "Class (index)",
            "Train Count",
            "Train %",
            "Val Count",
            "Val %",
            "Test Count",
            "Test %",
        ]
    ]
    for i in range(num_classes):
        name = QWEN_STYLE_TAGS_V1[i]
        rows.append(
            [
                "{} ({})".format(name[:28], i),
                str(ct[i]),
                "{:.2f}".format(100.0 * ct[i] / nt if nt else 0),
                str(cv[i]),
                "{:.2f}".format(100.0 * cv[i] / nv if nv else 0),
                str(cs[i]),
                "{:.2f}".format(100.0 * cs[i] / ns if ns else 0),
            ]
        )
    rows.append(
        [
            "**Total**",
            str(nt),
            "100.00",
            str(nv),
            "100.00",
            str(ns),
            "100.00",
        ]
    )

    widths = [max(len(rows[r][c]) for r in range(len(rows))) for c in range(len(rows[0]))]
    lines = []
    for ri, row in enumerate(rows):
        line = "|" + "|".join(" {} ".format(row[c].ljust(widths[c])) for c in range(len(row))) + "|"
        lines.append(line)
        if ri == 0:
            sep = "|" + "|".join(" {} ".format("-" * widths[c]) for c in range(len(row))) + "|"
            lines.append(sep)
    return "\n".join(lines)


def _json_id(x: str) -> Any:
    if x.isdigit():
        return int(x)
    try:
        return int(x, 10)
    except ValueError:
        return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified style split -> fashion_style_splits.json")
    ap.add_argument(
        "--input",
        default="output/fashion_style_adapter/fashion_style_mapping.json",
        help="Path to fashion_style_mapping.json",
    )
    ap.add_argument(
        "--output",
        default="output/fashion_style_adapter/fashion_style_splits.json",
        help="Output splits JSON",
    )
    ap.add_argument("--train-frac", type=float, default=0.7, help="Fraction for train (default 0.7)")
    ap.add_argument("--val-size", type=int, default=5000, help="Val size sampled from temp 30%% (default 5000)")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    inp = os.path.abspath(os.path.expanduser(args.input))
    outp = os.path.abspath(os.path.expanduser(args.output))

    ids, y = load_labeled_pairs(inp)
    id_to_label = {ids[i]: int(y[i]) for i in range(len(ids))}

    train_ids, val_ids, test_ids = split_train_temp_val_test(
        ids,
        y,
        train_frac=float(args.train_frac),
        val_size=int(args.val_size),
        random_state=int(args.random_state),
    )

    out_obj = {
        "train": [_json_id(i) for i in train_ids],
        "val": [_json_id(i) for i in val_ids],
        "test": [_json_id(i) for i in test_ids],
    }

    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    # Global distribution (labeled pool) for reference
    num_classes = len(QWEN_STYLE_TAGS_V1)
    global_c = _counts_for_split(ids, id_to_label, num_classes)
    ng = len(ids)

    print("\n## Labeled pool (before split)\n")
    print("| Class | Count | Global % |")
    print("| --- | ---: | ---: |")
    for i in range(num_classes):
        print("| {} ({}) | {} | {:.2f} |".format(QWEN_STYLE_TAGS_V1[i][:32], i, global_c[i], 100.0 * global_c[i] / ng))
    print("| **Total** | {} | 100.00 |\n".format(ng))

    print("## Train / Val / Test — per-split counts & percentages\n")
    print(_markdown_table(train_ids, val_ids, test_ids, id_to_label))
    print("\nSaved: {}\n".format(outp))


if __name__ == "__main__":
    main()
