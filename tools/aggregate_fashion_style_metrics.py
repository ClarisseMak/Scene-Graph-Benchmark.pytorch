#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Batch style metrics + plots for 8 eval runs (2 full + 6 ablation): reads each
OUTPUT_DIR/inference/<TEST dataset>/eval_results.pytorch and aggregates Top-1 / macro P/R/F1.

Usage (from repo root):
  python tools/aggregate_fashion_style_metrics.py \\
    --config-file configs/e2e_relation_fashion_finetune_R101_3060.yaml
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


def _setup_path():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


REPO = _setup_path()

from maskrcnn_benchmark.config import cfg, coerce_yacs_cli_opts  # noqa: E402
from maskrcnn_benchmark.data import make_data_loader  # noqa: E402

_calc_path = os.path.join(REPO, "tools", "calculate_style_metrics.py")
_spec = importlib.util.spec_from_file_location("_calc_style_metrics", _calc_path)
_calc_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_calc_mod)
run_style_metrics_from_eval_results = _calc_mod.run_style_metrics_from_eval_results


# Order: 2 full baseline + 6 ablation (matches full eval + run_fashion_ablation_tests.sh)
DEFAULT_RUNS: List[Tuple[str, str]] = [
    ("full_none", "output/eval_fashion_ablation_full_none"),
    ("full_tde", "output/eval_fashion_ablation_full_tde"),
    ("no_sc_none", "output/eval_fashion_ablation_no_sc_none"),
    ("no_sc_tde", "output/eval_fashion_ablation_no_sc_tde"),
    ("no_ec_none", "output/eval_fashion_ablation_no_ec_none"),
    ("no_ec_tde", "output/eval_fashion_ablation_no_ec_tde"),
    ("no_gc_none", "output/eval_fashion_ablation_no_gc_none"),
    ("no_gc_tde", "output/eval_fashion_ablation_no_gc_tde"),
]


def _inference_eval_path(repo_root: str, output_rel: str, dataset_name: str) -> str:
    return os.path.join(
        repo_root,
        output_rel,
        "inference",
        dataset_name,
        "eval_results.pytorch",
    )


def _plot_overview_bars(
    out_dir: str,
    labels: List[str],
    top1: List[float],
    macro_f1: List[float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        print("matplotlib unavailable: %s" % ex)
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.9), 8), sharex=True)
    axes[0].bar(x, top1, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Style classification — all completed runs")
    for i, v in enumerate(top1):
        axes[0].text(i, min(v + 0.02, 1.0), "%.3f" % v, ha="center", va="bottom", fontsize=7)

    axes[1].bar(x, macro_f1, color="seagreen", edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    for i, v in enumerate(macro_f1):
        axes[1].text(i, min(v + 0.02, 1.0), "%.3f" % v, ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "style_metrics_all_runs_bars.png"), dpi=150)
    plt.close(fig)


def _plot_grouped_none_vs_tde(
    out_dir: str,
    top1_map: Dict[str, float],
    macro_f1_map: Dict[str, float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        print("matplotlib grouped plots: %s" % ex)
        return

    g_names = ["full", "no_sc", "no_ec", "no_gc"]
    w = 0.35
    gx = np.arange(len(g_names))

    def pair(g: str):
        a = top1_map.get("%s_none" % g)
        b = top1_map.get("%s_tde" % g)
        return (
            float(a) if a is not None else float("nan"),
            float(b) if b is not None else float("nan"),
        )

    g_none = [pair(g)[0] for g in g_names]
    g_tde = [pair(g)[1] for g in g_names]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(gx - w / 2, g_none, width=w, label="EFFECT none", color="steelblue", edgecolor="black", linewidth=0.5)
    ax2.bar(gx + w / 2, g_tde, width=w, label="EFFECT TDE", color="coral", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(gx)
    ax2.set_xticklabels(g_names)
    ax2.set_ylabel("Top-1 accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.set_title("Top-1 by group (none vs TDE)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "style_metrics_grouped_none_vs_tde_top1.png"), dpi=150)
    plt.close(fig2)

    def pair_f1(g: str):
        a = macro_f1_map.get("%s_none" % g)
        b = macro_f1_map.get("%s_tde" % g)
        return (
            float(a) if a is not None else float("nan"),
            float(b) if b is not None else float("nan"),
        )

    g_none_f1 = [pair_f1(g)[0] for g in g_names]
    g_tde_f1 = [pair_f1(g)[1] for g in g_names]

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(gx - w / 2, g_none_f1, width=w, label="EFFECT none", color="steelblue", edgecolor="black", linewidth=0.5)
    ax3.bar(gx + w / 2, g_tde_f1, width=w, label="EFFECT TDE", color="coral", edgecolor="black", linewidth=0.5)
    ax3.set_xticks(gx)
    ax3.set_xticklabels(g_names)
    ax3.set_ylabel("Macro F1")
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.set_title("Macro F1 by group (none vs TDE)")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "style_metrics_grouped_none_vs_tde_macro_f1.png"), dpi=150)
    plt.close(fig3)


def main():
    parser = argparse.ArgumentParser(description="Aggregate style metrics across full + ablation eval runs")
    parser.add_argument("--config-file", required=True, help="Same YAML as relation_test_net")
    parser.add_argument(
        "--repo-root",
        default=REPO,
        help="Repo root (default: parent of tools/)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Write summary CSV/JSON/PNGs here (default: <repo>/output/style_metrics_aggregate)",
    )
    parser.add_argument(
        "--write-per-run",
        action="store_true",
        help="Also write style_metrics.txt / confusion matrix PNGs into each inference folder",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    out_dir = args.out_dir or os.path.join(repo_root, "output", "style_metrics_aggregate")
    os.makedirs(out_dir, exist_ok=True)

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(coerce_yacs_cli_opts(args.opts))
    cfg.freeze()

    dataset_names = list(cfg.DATASETS.TEST)
    if cfg.DATASETS.TO_TEST == "train":
        dataset_names = list(cfg.DATASETS.TRAIN)
    elif cfg.DATASETS.TO_TEST == "val":
        dataset_names = list(cfg.DATASETS.VAL)
    ds_name = dataset_names[0] if dataset_names else "VG_stanford_filtered_with_attribute_test"

    loaders = make_data_loader(
        cfg=cfg, mode="test", is_distributed=False, dataset_to_test=cfg.DATASETS.TO_TEST
    )
    dataset = loaders[0].dataset

    results: Dict[str, dict] = {}
    rows: List[dict] = []

    print("=" * 80)
    print("Style metrics aggregate (dataset: %s)" % ds_name)
    print("=" * 80)

    for run_key, rel_out in DEFAULT_RUNS:
        eval_path = _inference_eval_path(repo_root, rel_out, ds_name)
        inf_folder = os.path.dirname(eval_path)
        if not os.path.isfile(eval_path):
            print("\n[%s] SKIP (missing %s)" % (run_key, eval_path))
            continue

        print("\n--- %s ---\n" % run_key)
        m = run_style_metrics_from_eval_results(
            cfg,
            inf_folder,
            dataset,
            logger=None,
            eval_results_path=eval_path,
            write_artifacts=args.write_per_run,
        )
        if m is None:
            print("[%s] No valid style predictions." % run_key)
            continue
        results[run_key] = m
        rows.append(
            {
                "run": run_key,
                "top1_accuracy": m["top1_accuracy"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
                "num_evaluated": m["num_evaluated"],
            }
        )

    if not rows:
        print("No metrics computed; check eval_results.pytorch paths.")
        return 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    hdr = "{:16s} {:>10s} {:>10s} {:>10s} {:>10s} {:>8s}".format(
        "run", "Top-1", "mPrec", "mRec", "mF1", "N"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            "{:16s} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:8d}".format(
                r["run"],
                r["top1_accuracy"],
                r["macro_precision"],
                r["macro_recall"],
                r["macro_f1"],
                r["num_evaluated"],
            )
        )

    csv_path = os.path.join(out_dir, "style_metrics_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote %s" % csv_path)

    json_path = os.path.join(out_dir, "style_metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("Wrote %s" % json_path)

    top1_map = {r["run"]: r["top1_accuracy"] for r in rows}
    macro_map = {r["run"]: r["macro_f1"] for r in rows}

    labels: List[str] = []
    top1_list: List[float] = []
    macro_list: List[float] = []
    for run_key, _ in DEFAULT_RUNS:
        if run_key not in results:
            continue
        m = results[run_key]
        labels.append(run_key.replace("_", "\n"))
        top1_list.append(float(m["top1_accuracy"]))
        macro_list.append(float(m["macro_f1"]))

    if labels:
        _plot_overview_bars(out_dir, labels, top1_list, macro_list)
        print("Wrote %s" % os.path.join(out_dir, "style_metrics_all_runs_bars.png"))

    _plot_grouped_none_vs_tde(out_dir, top1_map, macro_map)
    print("Wrote grouped PNGs under %s" % out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
