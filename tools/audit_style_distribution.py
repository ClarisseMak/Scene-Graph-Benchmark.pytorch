#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Audit fashion style label distribution from DATASETS.STYLE_ANNOTATION_FILE JSON.

- Counts per class (0..9) using QWEN_STYLE_TAGS_V1 names
- Flags MISSING (0 in JSON) vs rare
- Optional: raw-tag vs stored label mismatch (full adapter JSON with images.*.style)
- Optional: image_ids present in JSON but absent from VG train split (not used in training)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# Repo root on path when run as python tools/audit_style_distribution.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from maskrcnn_benchmark.data.datasets.style_tags import (  # noqa: E402
    QWEN_STYLE_TAGS_V1,
    map_raw_tag_to_label,
)
from maskrcnn_benchmark.data.datasets.visual_genome import _load_style_annotation_map  # noqa: E402


NUM_STYLE = len(QWEN_STYLE_TAGS_V1)

# Rarest classes in v2 ontology (by typical frequency) — sample up to N paths each
RARE_FOCUS_INDICES = [9, 8, 7, 6, 4]  # Maximalist, Industrial, Minimalist, Luminous, Urban
MAX_SAMPLES_PER_RARE_CLASS = 20

OUTPUT_REL_PATH = os.path.join("output", "rare_style_samples_to_inspect.txt")


def _resolve_path(cfg_path: str, json_rel: str) -> str:
    if not json_rel:
        return ""
    p = os.path.expanduser(json_rel)
    if os.path.isabs(p):
        return p
    # relative to cwd (same as training)
    return os.path.abspath(os.path.join(os.getcwd(), p))


def _load_full_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_from_parsed(
    data: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, str], List[Tuple[str, str, int, int, str]]]:
    """
    Returns:
      id_to_label: image_id -> style_label
      id_to_path: image_id -> filesystem path if known
      mismatches: list of (image_id, tag_raw_used, expected_label, stored_label, note)
    """
    id_to_label: Dict[str, int] = {}
    id_to_path: Dict[str, str] = {}
    mismatches: List[Tuple[str, str, int, int, str]] = []

    by = data.get("by_image_id") if isinstance(data, dict) else None
    if isinstance(by, dict):
        for k, v in by.items():
            iid = str(k)
            if isinstance(v, dict):
                lab = int(v.get("style_label", -1))
            else:
                lab = int(v)
            id_to_label[iid] = lab

    images = data.get("images") if isinstance(data, dict) else None
    if isinstance(images, dict):
        for _, rec in images.items():
            if not isinstance(rec, dict):
                continue
            iid = rec.get("image_id")
            if iid is None:
                continue
            iid = str(iid)
            st = rec.get("style") or {}
            path = rec.get("path")
            if path:
                id_to_path[iid] = str(path)
            tag_raw = st.get("tag_raw_used")
            if tag_raw is not None and str(tag_raw).strip():
                expected = map_raw_tag_to_label(str(tag_raw))
                stored = st.get("style_label")
                if stored is not None:
                    stored_i = int(stored)
                    if expected != stored_i:
                        mismatches.append(
                            (iid, str(tag_raw), expected, stored_i, "tag_raw_used vs style.style_label")
                        )
            # Prefer images[] label if by_image_id missing this id
            if iid not in id_to_label and st.get("style_label") is not None:
                id_to_label[iid] = int(st["style_label"])

    if not id_to_label and isinstance(data, dict):
        # Fallback: only images, no by_image_id
        for _, rec in (images or {}).items():
            if not isinstance(rec, dict):
                continue
            iid = rec.get("image_id")
            if iid is None:
                continue
            iid = str(iid)
            st = rec.get("style") or {}
            if st.get("style_label") is not None:
                id_to_label[iid] = int(st["style_label"])
            p = rec.get("path")
            if p:
                id_to_path[iid] = str(p)

    return id_to_label, id_to_path, mismatches


def _by_vs_images_label_clash(data: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """If full JSON has both by_image_id and images, detect same image_id with different style_label."""
    out: List[Tuple[str, int, int]] = []
    by = data.get("by_image_id")
    im = data.get("images")
    if not isinstance(by, dict) or not isinstance(im, dict):
        return out
    for _, rec in im.items():
        if not isinstance(rec, dict):
            continue
        iid = rec.get("image_id")
        if iid is None:
            continue
        iid = str(iid)
        st = rec.get("style") or {}
        sl_img = st.get("style_label")
        if sl_img is None:
            continue
        ent = by.get(iid)
        if ent is None:
            continue
        sl_by = int(ent["style_label"]) if isinstance(ent, dict) else int(ent)
        a, b = int(sl_img), sl_by
        if a != b:
            out.append((iid, a, b))
    return out


def _guess_path(iid: str, img_dir: Optional[str]) -> str:
    if not img_dir:
        return ""
    return os.path.join(os.path.expanduser(img_dir), "{}.jpg".format(iid))


def _train_image_ids(cfg) -> Optional[Set[str]]:
    """VG train split image_id strings (same filtering as VGDataset), or None if skipped/failed."""
    try:
        import copy

        from maskrcnn_benchmark.data import datasets as D
        from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
    except Exception:
        return None

    if not cfg.DATASETS.TRAIN:
        return None
    name = cfg.DATASETS.TRAIN[0]
    try:
        data = DatasetCatalog.get(name, cfg)
        factory = getattr(D, data["factory"])
        args = copy.deepcopy(data["args"])
        if "capgraphs_file" in args:
            del args["capgraphs_file"]
        args["transforms"] = None
        ds = factory(**args)
        out = set()
        for info in ds.img_info:
            iid = info.get("image_id")
            if iid is not None:
                out.add(str(iid))
        return out
    except Exception:
        return None


def _markdown_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(rows[i][c]) for i in range(len(rows))) for c in range(len(rows[0]))]
    sep = "|" + "|".join(" {} ".format("-" * w) for w in widths) + "|"
    lines = []
    for ri, row in enumerate(rows):
        line = "|" + "|".join(" {:{}} ".format(row[c], widths[c]) for c in range(len(row))) + "|"
        lines.append(line)
        if ri == 0:
            lines.append(sep)
    return "\n".join(lines)


def run_audit(
    json_path: str,
    img_dir: Optional[str],
    train_ids: Optional[Set[str]],
    out_file: str,
) -> None:
    if not json_path or not os.path.isfile(json_path):
        print("ERROR: style JSON not found: {}".format(json_path))
        sys.exit(1)

    data = _load_full_json(json_path)
    id_to_label, id_to_path, mismatches = _collect_from_parsed(data if isinstance(data, dict) else {})

    # If compact file only, _load_style_annotation_map still works for counts
    if not id_to_label:
        m = _load_style_annotation_map(json_path)
        if m:
            id_to_label = dict(m)

    if not id_to_label:
        print("ERROR: Could not parse any image_id -> style_label from JSON.")
        sys.exit(1)

    counts = Counter()
    invalid = 0
    ignored_neg = 0
    for iid, lab in id_to_label.items():
        if lab < 0:
            ignored_neg += 1
            continue
        if lab >= NUM_STYLE:
            invalid += 1
            continue
        counts[lab] += 1

    total_labeled = sum(counts.values())
    total_keys = len(id_to_label)

    # --- Console: distribution table ---
    table_rows: List[List[str]] = [
        ["Class", "Index", "Count", "Percent", "Notes"],
    ]
    for idx in range(NUM_STYLE):
        name = QWEN_STYLE_TAGS_V1[idx]
        c = counts[idx]
        pct = 100.0 * c / total_labeled if total_labeled else 0.0
        if c == 0:
            note = "**MISSING IN DATASET** (no images with this label in JSON)"
        elif c < max(50, total_labeled // 200):
            note = "rare"
        else:
            note = ""
        table_rows.append([name, str(idx), str(c), "{:.2f}%".format(pct), note])

    print("\n## Style distribution (labeled entries: {}, style_label<0 ignored: {}, out-of-range: {})\n".format(
        total_labeled, ignored_neg, invalid,
    ))
    print("Total keys in map: {}\n".format(total_keys))
    print(_markdown_table(table_rows))

    # --- Mismatch / training discrepancy ---
    print("\n## Mapping quality\n")
    clash = _by_vs_images_label_clash(data if isinstance(data, dict) else {})
    if clash:
        print(
            "- **by_image_id vs images[] label clash**: **{}** ids (same image, different stored labels)".format(
                len(clash)
            )
        )
        for row in clash[:10]:
            print("  - image_id={}  images.style={}  by_image_id.style_label={}".format(row[0], row[1], row[2]))
        if len(clash) > 10:
            print("  ... and {} more".format(len(clash) - 10))

    if mismatches:
        print(
            "- **Label mismatch** (recomputed from `tag_raw_used` vs stored `style_label`): **{}** images".format(
                len(mismatches)
            )
        )
        show = mismatches[:15]
        for t in show:
            print("  - image_id={} raw={!r} expected={} stored={} ({})".format(t[0], t[1], t[2], t[3], t[4]))
        if len(mismatches) > 15:
            print("  ... and {} more".format(len(mismatches) - 15))
    else:
        print("- No `images[].style.tag_raw_used` vs `style_label` mismatches detected (compact JSON or no raw tags).")

    if train_ids is not None:
        in_train = 0
        not_train = 0
        for iid in id_to_label:
            if id_to_label[iid] < 0 or id_to_label[iid] >= NUM_STYLE:
                continue
            if iid in train_ids:
                in_train += 1
            else:
                not_train += 1
        print(
            "\n- **Train split coverage** (cfg.DATASETS.TRAIN[0]): labeled images **in train**: {}, **not in train**: {}".format(
                in_train, not_train
            )
        )
        if not_train:
            print(
                "  (Labels in JSON for ids outside the filtered train split are **not used in style_loss** during training.)"
            )
        # Per-class not-in-train for rare focus
        for idx in RARE_FOCUS_INDICES:
            cls_ids = [i for i, l in id_to_label.items() if l == idx]
            nt = sum(1 for i in cls_ids if i not in train_ids)
            if cls_ids and nt:
                print(
                    "  - class {} ({}): {} / {} samples **not** in train split".format(
                        idx, QWEN_STYLE_TAGS_V1[idx], nt, len(cls_ids)
                    )
                )
    else:
        print("\n- Train split coverage: *(skipped — use --train-coverage to enable)*")

    # --- Write rare sample file ---
    os.makedirs(os.path.dirname(os.path.abspath(out_file)) or ".", exist_ok=True)
    lines_out: List[str] = []
    lines_out.append("# Rare / focus style classes — samples for manual inspection")
    lines_out.append("# JSON: {}".format(json_path))
    lines_out.append("")

    for idx in RARE_FOCUS_INDICES:
        name = QWEN_STYLE_TAGS_V1[idx]
        members = [iid for iid, lab in id_to_label.items() if lab == idx]
        members.sort(key=lambda x: int(x) if str(x).isdigit() else x)
        lines_out.append("## {} (index {})\n".format(name, idx))
        if not members:
            lines_out.append("STATUS: MISSING IN DATASET (count=0 in JSON)\n")
            continue
        lines_out.append("COUNT_IN_JSON: {}\n".format(len(members)))
        lines_out.append("image_id\tpath_or_guess\n")
        for iid in members[:MAX_SAMPLES_PER_RARE_CLASS]:
            p = id_to_path.get(iid) or _guess_path(iid, img_dir)
            lines_out.append("{}\t{}".format(iid, p or "(no path; set --img-dir to suggest VG_100K/{{id}}.jpg)"))
        if len(members) > MAX_SAMPLES_PER_RARE_CLASS:
            lines_out.append("... ({} more ids omitted; increase MAX_SAMPLES_PER_RARE_CLASS in script if needed)".format(
                len(members) - MAX_SAMPLES_PER_RARE_CLASS
            ))
        lines_out.append("")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    print("\n## Output file\n\nSaved: **{}**\n".format(os.path.abspath(out_file)))


def main():
    parser = argparse.ArgumentParser(description="Audit style label distribution from style JSON")
    parser.add_argument("--config-file", default="", help="YAML config (reads DATASETS.STYLE_ANNOTATION_FILE)")
    parser.add_argument(
        "--style-json",
        default="",
        help="Override path to style JSON (default: from config)",
    )
    parser.add_argument(
        "--img-dir",
        default="",
        help="Optional VG image folder; used to guess paths as {{img_dir}}/{{image_id}}.jpg when JSON has no path",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_REL_PATH,
        help="Output txt path (default: output/rare_style_samples_to_inspect.txt)",
    )
    parser.add_argument(
        "--train-coverage",
        action="store_true",
        help="Load VG train split (cfg.DATASETS.TRAIN[0]) and report JSON ids not in train",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    json_path = args.style_json
    img_dir = args.img_dir or None
    train_ids: Optional[Set[str]] = None

    if args.config_file:
        from maskrcnn_benchmark.config import cfg, coerce_yacs_cli_opts

        cfg.merge_from_file(args.config_file)
        if args.opts:
            cfg.merge_from_list(coerce_yacs_cli_opts(args.opts))
        cfg.freeze()
        if not json_path:
            rel = getattr(cfg.DATASETS, "STYLE_ANNOTATION_FILE", "") or ""
            json_path = _resolve_path(args.config_file, rel)
        if not img_dir and args.train_coverage:
            # Best-effort: DATA_DIR + catalog img_dir from TRAIN dataset
            try:
                import copy

                from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog

                name = cfg.DATASETS.TRAIN[0]
                p = name.rfind("_")
                base = name[:p]
                attrs = DatasetCatalog.DATASETS[base]
                data_dir = DatasetCatalog.DATA_DIR
                img_dir = os.path.join(data_dir, attrs["img_dir"])
            except Exception:
                pass
        if args.train_coverage:
            train_ids = _train_image_ids(cfg)

    if not json_path:
        print("ERROR: Provide --config-file or --style-json")
        sys.exit(1)

    run_audit(json_path, img_dir, train_ids, args.output)


if __name__ == "__main__":
    main()
