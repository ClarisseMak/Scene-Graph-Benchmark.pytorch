"""
Convert Qwen3-VL image-level style JSON + optional SGDet outputs into:
  - Per-image style_label (0..9) aligned with style_tags.QWEN_STYLE_TAGS_V1 (10-class v2 ontology)
  - Optional predicate_init from detected boxes (30-class fashion-context ontology)
  - Mapping files consumable by VGDataset (same predicate order as fashion_predicates.py)
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Repo root on path for imports
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from maskrcnn_benchmark.data.datasets.fashion_predicates import fashion_context_predicates_v1
from maskrcnn_benchmark.data.datasets.style_tags import QWEN_STYLE_TAGS_V1, map_raw_tag_to_label


def _box_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, b[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _area(b: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = _area((ix1, iy1, ix2, iy2))
    if inter <= 0:
        return 0.0
    ua = _area(a) + _area(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _contains(outer: Tuple[float, float, float, float], inner: Tuple[float, float, float, float], tol: float = 0.0) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return (ix1 >= ox1 - tol) and (iy1 >= oy1 - tol) and (ix2 <= ox2 + tol) and (iy2 <= oy2 + tol)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def default_fashion_predicates_30() -> List[str]:
    return fashion_context_predicates_v1()


def _is_person(label: str) -> bool:
    s = (label or "").lower()
    return s == "person" or "person" in s or s in {"man", "woman", "boy", "girl"}


def _is_fashion_item(label: str) -> bool:
    s = (label or "").lower()
    keys = [
        "shirt", "t-shirt", "tshirt", "sweater", "hoodie", "jacket", "coat",
        "pants", "trousers", "jeans", "skirt", "shorts", "dress",
        "shoe", "sneaker", "boot", "heel",
        "bag", "backpack", "handbag",
        "hat", "cap",
        "glasses", "sunglasses",
        "belt", "scarf", "watch", "necklace", "bracelet", "earring",
    ]
    return any(k in s for k in keys)


def infer_predicate(
    subj_label: str,
    obj_label: str,
    subj_box: Tuple[float, float, float, float],
    obj_box: Tuple[float, float, float, float],
    img_diag: float,
) -> str:
    if _is_person(subj_label) and _is_fashion_item(obj_label):
        return "wearing"
    i = _iou(subj_box, obj_box)
    if i >= 0.35:
        return "overlapping"
    if _contains(subj_box, obj_box, tol=2.0):
        return "surrounding"
    if _contains(obj_box, subj_box, tol=2.0):
        return "inside"
    sx, sy = _center(subj_box)
    ox, oy = _center(obj_box)
    dx, dy = ox - sx, oy - sy
    if img_diag > 0:
        if _dist((sx, sy), (ox, oy)) / img_diag <= 0.18:
            return "next-to"
    if abs(dx) >= abs(dy):
        return "right-of" if dx > 0 else "left-of"
    return "below" if dy > 0 else "above"


def build_relations_for_image(
    boxes_xyxy: List[List[float]],
    labels: List[int],
    ind_to_classes: List[str],
    predicates: List[str],
    max_pairs: int = 2000,
) -> Dict[str, Any]:
    n = min(len(boxes_xyxy), len(labels))
    if n <= 1:
        return {"rel_pairs": [], "rel_labels": [], "rel_labels_name": []}

    boxes = [_box_xyxy(b) for b in boxes_xyxy[:n]]
    cls = [ind_to_classes[int(i)] if int(i) < len(ind_to_classes) else str(int(i)) for i in labels[:n]]

    max_x2 = max(b[2] for b in boxes)
    max_y2 = max(b[3] for b in boxes)
    img_diag = math.sqrt(max_x2 * max_x2 + max_y2 * max_y2) if max_x2 > 0 and max_y2 > 0 else 1.0

    pred_to_idx = {p: i for i, p in enumerate(predicates)}

    rel_pairs: List[List[int]] = []
    rel_labels: List[int] = []
    rel_names: List[str] = []

    count = 0
    for s in range(n):
        for o in range(n):
            if s == o:
                continue
            name = infer_predicate(cls[s], cls[o], boxes[s], boxes[o], img_diag)
            if name not in pred_to_idx:
                continue
            rel_pairs.append([s, o])
            rel_labels.append(int(pred_to_idx[name]))
            rel_names.append(name)
            count += 1
            if count >= max_pairs:
                break
        if count >= max_pairs:
            break

    return {"rel_pairs": rel_pairs, "rel_labels": rel_labels, "rel_labels_name": rel_names}


def _compute_style_balance_sample_weights(
    by_image_id: Dict[str, Any],
    threshold: float,
    max_class_weight: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Per-image sampling weight for WeightedRandomSampler: boost classes with fraction < threshold of labeled set.
    weight_image = min(max_class_weight, (threshold * N) / n_class) for rare classes; else 1.0.
    """
    id_to_label: Dict[str, int] = {}
    for iid, v in by_image_id.items():
        lab = v.get("style_label", -1) if isinstance(v, dict) else v
        try:
            id_to_label[str(iid)] = int(lab)
        except (TypeError, ValueError):
            id_to_label[str(iid)] = -1

    labeled_ids = [i for i, lab in id_to_label.items() if lab >= 0]
    n_labeled = len(labeled_ids)
    num_classes = len(QWEN_STYLE_TAGS_V1)
    if n_labeled == 0:
        return {}, {
            "threshold_fraction": threshold,
            "labeled_image_count": 0,
            "class_counts": [0] * num_classes,
            "note": "no labeled images",
        }

    cnt = Counter(id_to_label[i] for i in labeled_ids)
    class_counts = [int(cnt.get(c, 0)) for c in range(num_classes)]

    weight_for_class: List[float] = []
    for c in range(num_classes):
        n_c = class_counts[c]
        frac = n_c / n_labeled
        if n_c <= 0:
            w = 1.0
        elif frac < threshold:
            w = min(max_class_weight, (threshold * n_labeled) / float(n_c))
        else:
            w = 1.0
        weight_for_class.append(float(w))

    train_weight_by_id: Dict[str, float] = {}
    for iid, lab in id_to_label.items():
        if lab < 0 or lab >= num_classes:
            train_weight_by_id[iid] = 1.0
        else:
            train_weight_by_id[iid] = weight_for_class[lab]

    meta = {
        "threshold_fraction": threshold,
        "max_class_weight": max_class_weight,
        "labeled_image_count": n_labeled,
        "class_counts": class_counts,
        "class_fractions": [class_counts[c] / n_labeled for c in range(num_classes)],
        "weight_per_class": {str(c): weight_for_class[c] for c in range(num_classes)},
        "class_names": list(QWEN_STYLE_TAGS_V1),
    }
    return train_weight_by_id, meta


def _primary_tag_en(record: Dict[str, Any]) -> Optional[str]:
    tags = record.get("tags_en") or []
    if isinstance(tags, list) and len(tags) > 0:
        return str(tags[0]).strip()
    t = record.get("tag_en")
    return str(t).strip() if t is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Qwen3-VL style -> style_label (0-9) + optional predicate init; "
                    "writes mapping files for VGDataset (predicate indices match fashion_context)."
    )
    ap.add_argument("--style_json", required=True, help="Path to datasets/vg/vg_style_annotations.json")
    ap.add_argument("--custom_data_info", default="", help="Path to DETECTED_SGG_DIR/custom_data_info.json")
    ap.add_argument("--custom_prediction", default="", help="Path to DETECTED_SGG_DIR/custom_prediction.json")
    ap.add_argument("--output", required=True, help="Main output json path")
    ap.add_argument(
        "--mapping_output",
        default="",
        help="Optional path for compact mapping-only json (default: <output_dir>/fashion_style_mapping.json)",
    )
    ap.add_argument("--max_pairs", type=int, default=2000, help="Max relations per image")
    ap.add_argument(
        "--no-balance-oversample",
        action="store_true",
        help="Disable train_sample_weight_by_image_id (no rare-class boost in mapping)",
    )
    ap.add_argument(
        "--balance-rare-threshold",
        type=float,
        default=0.05,
        help="Classes with count/N below this get boosted sampling weight (default 5%%)",
    )
    ap.add_argument(
        "--balance-max-class-weight",
        type=float,
        default=100.0,
        help="Cap on per-class oversampling multiplier",
    )
    args = ap.parse_args()

    style = json.load(open(args.style_json, "r", encoding="utf-8"))

    predicates = default_fashion_predicates_30()
    assert len(predicates) == 30, "Fashion-context ontology must be 30 entries incl. __background__"

    ind_to_classes: List[str] = ["__background__", "person"]
    pred_pack: Dict[str, Any] = {}

    if args.custom_data_info and os.path.exists(args.custom_data_info):
        info = json.load(open(args.custom_data_info, "r", encoding="utf-8"))
        ind_to_classes = info.get("ind_to_classes", ind_to_classes)
        predicates = info.get("ind_to_predicates", predicates)
        if len(predicates) != 30:
            print(
                "[WARN] ind_to_predicates length is %s; expected 30 for fashion_context. "
                "Using fashion_context_predicates_v1()." % len(predicates)
            )
            predicates = default_fashion_predicates_30()

    if args.custom_prediction and os.path.exists(args.custom_prediction):
        pred_pack = json.load(open(args.custom_prediction, "r", encoding="utf-8"))

    style_tag_to_idx = {t: i for i, t in enumerate(QWEN_STYLE_TAGS_V1)}

    out: Dict[str, Any] = {
        "meta": {
            "source": "qwen3-vl style tags + heuristic predicate init",
            "style_json": os.path.abspath(args.style_json),
            "predicate_set": "fashion_context",
            "num_predicates": len(predicates),
            "style_num_classes": len(QWEN_STYLE_TAGS_V1),
            "style_tag_to_idx": style_tag_to_idx,
            "idx_to_style_tag": list(QWEN_STYLE_TAGS_V1),
            "ind_to_predicates": predicates,
            "predicate_to_idx": {p: i for i, p in enumerate(predicates)},
        },
        "images": {},
        "by_image_id": {},
    }

    for k, v in style.items():
        image_id = v.get("image_id")
        if image_id is None:
            try:
                image_id = int(k)
            except Exception:
                image_id = k

        tag_raw = _primary_tag_en(v)
        style_label = map_raw_tag_to_label(tag_raw)
        canonical = QWEN_STYLE_TAGS_V1[style_label]

        rec: Dict[str, Any] = {
            "image_id": image_id,
            "path": v.get("path"),
            "style": {
                "tag_en": v.get("tag_en"),
                "tags_en": v.get("tags_en", []),
                "tag_display": v.get("tag_display"),
                "tags_display": v.get("tags_display", []),
                "tag_raw_used": tag_raw,
                "style_label": style_label,
                "style_tag_canonical": canonical,
            },
        }

        if pred_pack:
            det_item = None
            if isinstance(pred_pack, list):
                sgg_index = v.get("sgg_index", None)
                if isinstance(sgg_index, int) and 0 <= sgg_index < len(pred_pack):
                    det_item = pred_pack[sgg_index]
            elif isinstance(pred_pack, dict):
                det_item = pred_pack.get(str(image_id)) or pred_pack.get(k)

            if det_item and isinstance(det_item, dict) and "bbox" in det_item and "bbox_labels" in det_item:
                rel_init = build_relations_for_image(
                    boxes_xyxy=det_item.get("bbox", []),
                    labels=det_item.get("bbox_labels", []),
                    ind_to_classes=ind_to_classes,
                    predicates=predicates,
                    max_pairs=args.max_pairs,
                )
                rec["predicate_init"] = {
                    "predicates": predicates,
                    **rel_init,
                }

        out["images"][str(k)] = rec
        out["by_image_id"][str(image_id)] = {
            "style_label": style_label,
            "style_tag_canonical": canonical,
        }

    balance_meta: Optional[Dict[str, Any]] = None
    train_sample_weight_by_image_id: Optional[Dict[str, float]] = None
    if not args.no_balance_oversample:
        train_sample_weight_by_image_id, balance_meta = _compute_style_balance_sample_weights(
            out["by_image_id"],
            threshold=float(args.balance_rare_threshold),
            max_class_weight=float(args.balance_max_class_weight),
        )
        out["meta"]["style_balance_oversample"] = balance_meta
        out["train_sample_weight_by_image_id"] = train_sample_weight_by_image_id
        print("\n=== Style balance (train_sample_weight_by_image_id) ===")
        print("labeled_image_count:", balance_meta.get("labeled_image_count"))
        print("threshold_fraction:", balance_meta.get("threshold_fraction"))
        for c, name in enumerate(QWEN_STYLE_TAGS_V1):
            nc = balance_meta["class_counts"][c]
            fr = balance_meta["class_fractions"][c]
            w = balance_meta["weight_per_class"][str(c)]
            tag = ""
            if nc == 0:
                tag = "  [MISSING IN DATASET]"
            elif fr < args.balance_rare_threshold:
                tag = "  [RARE -> boosted]"
            print("  [%d] %-22s  n=%6d  frac=%.4f  sampler_w=%.3f%s" % (c, name[:22], nc, fr, w, tag))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    mapping_path = args.mapping_output or os.path.join(out_dir, "fashion_style_mapping.json")
    mapping_compact: Dict[str, Any] = {
        "meta": out["meta"],
        "by_image_id": out["by_image_id"],
    }
    if train_sample_weight_by_image_id is not None:
        mapping_compact["train_sample_weight_by_image_id"] = train_sample_weight_by_image_id
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_compact, f, ensure_ascii=False)

    print("Wrote:", os.path.abspath(args.output))
    print("Wrote:", os.path.abspath(mapping_path))


if __name__ == "__main__":
    main()
