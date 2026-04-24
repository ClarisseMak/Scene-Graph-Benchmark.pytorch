#!/usr/bin/env bash
# Sequential ablation/full runs for fashion finetune (relation_test_net).
# Usage (from repo root):
#   bash tools/run_fashion_ablation_tests.sh
#   CKPT=/path/to/model_final.pth bash tools/run_fashion_ablation_tests.sh
#   CUDA_VISIBLE_DEVICES=1 bash tools/run_fashion_ablation_tests.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${CKPT:-$ROOT/output/fashion_finetune_3060_hgcn/model_final.pth}"
CONFIG="${CONFIG:-configs/e2e_relation_fashion_finetune_R101_3060.yaml}"
PY="${PYTHON:-python}"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  echo "Set CKPT=/path/to/model_final.pth" >&2
  exit 1
fi

run_one() {
  local name="$1"
  local out_dir="$2"
  shift
  shift
  echo "========================================"
  echo "RUN: $name"
  echo "========================================"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" tools/relation_test_net.py \
    --config-file "$CONFIG" \
    MODEL.WEIGHT "$CKPT" \
    TEST.ALLOW_LOAD_FROM_CACHE false \
    "$@"
  echo "[metrics] calculating style metrics for $name ..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" tools/calculate_style_metrics.py \
    --config-file "$CONFIG" \
    OUTPUT_DIR "$out_dir"
  echo ""
}

# 8 组：full/no_sc/no_ec/no_gc × (none | TDE)
run_one "ablation_full_none" "$ROOT/output/eval_fashion_ablation_full_none" \
  OUTPUT_DIR "$ROOT/output/eval_fashion_ablation_full_none"

run_one "ablation_full_tde" "$ROOT/output/eval_fashion_ablation_full_tde" \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS true \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
  OUTPUT_DIR "$ROOT/output/eval_fashion_ablation_full_tde"

run_one "ablation_no_sc_none" "$ROOT/output/eval_fashion_ablation_no_sc_none" \
  TEST.ABLATION_MODE no_sc \
  OUTPUT_DIR "$ROOT/output/eval_fashion_ablation_no_sc_none"

run_one "ablation_no_sc_tde" "$ROOT/output/eval_fashion_ablation_no_sc_tde" \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS true \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
  TEST.ABLATION_MODE no_sc \
  OUTPUT_DIR "$ROOT/output/eval_fashion_ablation_no_sc_tde"

run_one "ablation_no_ec_none" "$ROOT/output/eval_fashion_ablation_no_ec_none" \
  TEST.ABLATION_MODE no_ec \
  OUTPUT_DIR "$ROOT/output/eval_fashion_ablation_no_ec_none"

run_one "ablation_no_ec_tde" "$ROOT/output/eval_fashion_ablation_no_ec_tde" \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS true \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
  TEST.ABLATION_MODE no_ec \
  OUTPUT_DIR "$ROOT/output/eval_fashion_ablation_no_ec_tde"

run_one "ablation_no_gc_none" "$ROOT/output/eval_fashion_ablation_no_gc_none" \
  TEST.ABLATION_MODE no_gc \
  OUTPUT_DIR "$ROOT/output/eval_fashion_ablation_no_gc_none"

run_one "ablation_no_gc_tde" "$ROOT/output/eval_fashion_ablation_no_gc_tde" \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS true \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
  TEST.ABLATION_MODE no_gc \
  OUTPUT_DIR "$ROOT/output/eval_fashion_ablation_no_gc_tde"

echo "All 8 runs (full + ablations) finished OK."
