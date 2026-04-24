# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Style classification metrics (Top-1 accuracy, confusion matrix, per-class P/R/F1)
aligned with common hierarchical-context recognition paper tables.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from maskrcnn_benchmark.data.datasets import style_tags


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def _per_class_prf1(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precision, recall, f1 per true-class row (one-vs-rest on confusion)."""
    num = cm.shape[0]
    prec = np.zeros(num, dtype=np.float64)
    rec = np.zeros(num, dtype=np.float64)
    f1 = np.zeros(num, dtype=np.float64)
    for c in range(num):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * prec[c] * rec[c] / (prec[c] + rec[c])) if (prec[c] + rec[c]) > 0 else 0.0
    return prec, rec, f1


def compute_style_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[List[str]] = None,
) -> dict:
    num_classes = len(class_names) if class_names is not None else max(max(y_true, default=0), max(y_pred, default=0)) + 1
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    cm = _confusion_matrix(yt, yp, num_classes)
    correct = int(np.diag(cm).sum())
    total = int(cm.sum())
    top1 = correct / total if total > 0 else 0.0
    prec, rec, f1 = _per_class_prf1(cm)
    macro_p = float(prec.mean()) if num_classes else 0.0
    macro_r = float(rec.mean()) if num_classes else 0.0
    macro_f1 = float(f1.mean()) if num_classes else 0.0
    return {
        "num_classes": num_classes,
        "top1_accuracy": top1,
        "num_evaluated": total,
        "confusion_matrix": cm,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "class_names": class_names or [str(i) for i in range(num_classes)],
    }


def format_metrics_table(metrics: dict) -> str:
    lines = []
    lines.append("Style classification (image-level)")
    lines.append("Top-1 Accuracy: {:.4f}  (N={})".format(metrics["top1_accuracy"], metrics["num_evaluated"]))
    lines.append("")
    lines.append("{:32s} {:>12s} {:>12s} {:>12s}".format("Class", "Precision", "Recall", "F1-score"))
    names = metrics["class_names"]
    for i, name in enumerate(names):
        if i >= len(metrics["precision"]):
            break
        lines.append(
            "{:32s} {:12.4f} {:12.4f} {:12.4f}".format(
                name[:32],
                metrics["precision"][i],
                metrics["recall"][i],
                metrics["f1"][i],
            )
        )
    lines.append("")
    lines.append(
        "Macro avg: {:>21s} {:12.4f} {:12.4f} {:12.4f}".format(
            "",
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_f1"],
        )
    )
    return "\n".join(lines)


def save_confusion_csv(path: str, cm: np.ndarray, class_names: List[str]) -> None:
    n = cm.shape[0]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [""] + class_names[:n]
        w.writerow(header)
        for i in range(n):
            row = [class_names[i] if i < len(class_names) else str(i)] + [int(cm[i, j]) for j in range(n)]
            w.writerow(row)


def maybe_plot_confusion(path: str, cm: np.ndarray, class_names: List[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names[:n], rotation=45, ha="right")
    ax.set_yticklabels(class_names[:n])
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", color="w" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_style_metrics_from_eval_results(
    cfg,
    output_folder: str,
    dataset,
    logger: Optional[logging.Logger] = None,
    eval_results_path: Optional[str] = None,
    write_artifacts: bool = True,
) -> Optional[dict]:
    """
    Load eval_results.pytorch from a relation_test_net run and compare pred_style to GT style_label.
    """
    path = eval_results_path or os.path.join(output_folder, "eval_results.pytorch")
    if not os.path.isfile(path):
        if logger:
            logger.warning("eval_results.pytorch not found at %s", path)
        return None
    blob = torch.load(path, map_location=torch.device("cpu"))
    predictions = blob.get("predictions", blob)
    if predictions is None:
        if logger:
            logger.warning("No predictions in %s", path)
        return None

    num_classes = int(getattr(cfg.MODEL.ROI_RELATION_HEAD, "STYLE_NUM_CLASSES", 10))
    names = list(style_tags.QWEN_STYLE_TAGS_V1)[:num_classes]
    while len(names) < num_classes:
        names.append("class_%d" % len(names))

    y_true: List[int] = []
    y_pred: List[int] = []

    n = min(len(predictions), len(dataset))
    for i in range(n):
        pred = predictions[i]
        if not pred.has_field("pred_style"):
            continue
        gt = dataset.get_groundtruth(i, evaluation=True)
        if not gt.has_field("style_label"):
            continue
        gt_lab = int(gt.get_field("style_label").item())
        if gt_lab < 0:
            continue
        pr_lab = int(pred.get_field("pred_style").item())
        y_true.append(gt_lab)
        y_pred.append(pr_lab)

    if not y_true:
        if logger:
            logger.warning("No style predictions with valid GT style_label; run inference with CausalAnalysisPredictor + style head.")
        return None

    metrics = compute_style_metrics(y_true, y_pred, class_names=names)
    text = format_metrics_table(metrics)
    if logger:
        logger.info("\n%s", text)
    else:
        print(text)

    out_folder = output_folder or os.path.dirname(path)
    if write_artifacts:
        out_txt = os.path.join(out_folder, "style_metrics.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text + "\n")

        out_csv = os.path.join(out_folder, "style_confusion_matrix.csv")
        save_confusion_csv(out_csv, metrics["confusion_matrix"], metrics["class_names"])

        out_png = os.path.join(out_folder, "style_confusion_matrix.png")
        maybe_plot_confusion(out_png, metrics["confusion_matrix"], metrics["class_names"])

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Style metrics from eval_results.pytorch + dataset")
    parser.add_argument("--config-file", required=True)
    parser.add_argument(
        "--eval-results",
        default="",
        help="Path to eval_results.pytorch (default: OUTPUT_DIR/inference/<first TEST dataset>/eval_results.pytorch)",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    from maskrcnn_benchmark.config import cfg, coerce_yacs_cli_opts
    from maskrcnn_benchmark.data import make_data_loader

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(coerce_yacs_cli_opts(args.opts))
    cfg.freeze()

    dataset_names = list(cfg.DATASETS.TEST)
    if cfg.DATASETS.TO_TEST == "train":
        dataset_names = list(cfg.DATASETS.TRAIN)
    elif cfg.DATASETS.TO_TEST == "val":
        dataset_names = list(cfg.DATASETS.VAL)

    loaders = make_data_loader(cfg=cfg, mode="test", is_distributed=False, dataset_to_test=cfg.DATASETS.TO_TEST)
    dataset = loaders[0].dataset

    if args.eval_results:
        eval_path = os.path.abspath(os.path.expanduser(args.eval_results))
        out_folder = os.path.dirname(eval_path)
    else:
        ds_name = dataset_names[0] if dataset_names else "test"
        out_folder = os.path.join(cfg.OUTPUT_DIR, "inference", ds_name)
        eval_path = os.path.join(out_folder, "eval_results.pytorch")

    run_style_metrics_from_eval_results(cfg, out_folder, dataset, eval_results_path=eval_path)


if __name__ == "__main__":
    main()
