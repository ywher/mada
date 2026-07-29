#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-cityscapes2acdc}"
GPU="${GPU:-0}"
CONDA_ENV="${CONDA_ENV:-reinpy10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/madav2_smoke_${USER}}"
MODELS="${MODELS:-deeplab101 dinov3_base_rein_hrda}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MADAV2_RESNET101_PRETRAINED="${REPO_ROOT}/pretrained/resnet101-5d3b4d8f.pth"

for model in ${MODELS}; do
  ratio="1_64"
  if [[ "${DATASET}" == "cityscapes2mapillary" ]]; then
    ratio="1_128"
  fi
  experiment_dir="${OUTPUT_ROOT}/${DATASET}/${ratio}/${model}"
  conda run -n "${CONDA_ENV}" python tools/prepare_experiment.py \
    --dataset "${DATASET}" \
    --model "${model}" \
    --output-root "${OUTPUT_ROOT}" \
    --max-iters 2 \
    --stage1-iters 1 \
    --source-iters 1 \
    --workers 0
  conda run -n "${CONDA_ENV}" python tools/smoke_train.py \
    --config "${experiment_dir}/source.yml" \
    --checkpoint "${experiment_dir}/smoke/model_smoke.pth" \
    --device cuda:0
  conda run -n "${CONDA_ENV}" python tools/evaluate_checkpoints.py \
    --config "${experiment_dir}/source.yml" \
    --run-dir "${experiment_dir}/smoke" \
    --checkpoints model_smoke.pth \
    --max-samples 2 \
    --device cuda:0
done

echo "MADAv2 training/evaluation smoke test passed."
