#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:?Set DATASET, e.g. gta2cityscapes}"
MODEL="${MODEL:-dinov3_base_rein_hrda}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/controlled}"
MAX_ITERS="${MAX_ITERS:-40000}"
STAGE1_ITERS="${STAGE1_ITERS:-20000}"
SOURCE_ITERS="${SOURCE_ITERS:-40000}"
WORKERS="${WORKERS:-4}"
CONDA_ENV="${CONDA_ENV:-reinpy10}"

RATIO="1_64"
if [[ "${DATASET}" == "cityscapes2mapillary" ]]; then
  RATIO="1_128"
fi
RATIO="${RATIO_NAME:-${RATIO}}"
BUDGET_ARGS=()
if [[ -n "${BUDGET:-}" ]]; then
  BUDGET_ARGS+=(--budget "${BUDGET}")
fi
EXP_DIR="${OUTPUT_ROOT}/${DATASET}/${RATIO}/${MODEL}"

conda run -n "${CONDA_ENV}" python tools/prepare_experiment.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-iters "${MAX_ITERS}" \
  --stage1-iters "${STAGE1_ITERS}" \
  --source-iters "${SOURCE_ITERS}" \
  --workers "${WORKERS}" \
  --ratio-name "${RATIO}" \
  "${BUDGET_ARGS[@]}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MADAV2_RESNET101_PRETRAINED="${REPO_ROOT}/pretrained/resnet101-5d3b4d8f.pth"

if [[ ! -f "${EXP_DIR}/source/model_best.pth" ]]; then
  conda run -n "${CONDA_ENV}" python train_source_warmup.py \
    --config "${EXP_DIR}/source.yml" \
    --run-dir "${EXP_DIR}/source" \
    --device cuda:0
fi

if [[ ! -f "${EXP_DIR}/selection/selected_images.txt" ]]; then
  conda run -n "${CONDA_ENV}" python tools/madav2_acquisition.py \
    --config "${EXP_DIR}/acquisition.yml" \
    --phase initial \
    --device cuda:0
fi

if [[ ! -f "${EXP_DIR}/stage1/model_best.pth" ]]; then
  conda run -n "${CONDA_ENV}" python step1_train_active_sup_only.py \
    --config "${EXP_DIR}/stage1.yml" \
    --run-dir "${EXP_DIR}/stage1"
fi

if [[ ! -f "${EXP_DIR}/selection/target_stage1_centroids.npy" ]]; then
  conda run -n "${CONDA_ENV}" python tools/madav2_acquisition.py \
    --config "${EXP_DIR}/acquisition.yml" \
    --phase post-stage1 \
    --device cuda:0
fi

if [[ ! -f "${EXP_DIR}/stage2/model_best.pth" ]]; then
  conda run -n "${CONDA_ENV}" python step2_train_active_semi_sup.py \
    --config "${EXP_DIR}/stage2.yml" \
    --run-dir "${EXP_DIR}/stage2"
fi
