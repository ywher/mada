#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-reinpy10}"
CREATE_ENV="${CREATE_ENV:-0}"
DATA_ROOT="${DATA_ROOT:-}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-}"

if [[ "${CREATE_ENV}" == "1" ]]; then
  TARGET_ENV="${TARGET_ENV:-madav2}"
  if ! conda env list | awk '{print $1}' | grep -qx "${TARGET_ENV}"; then
    conda create -y -n "${TARGET_ENV}" --clone "${CONDA_ENV}"
  fi
  CONDA_ENV="${TARGET_ENV}"
fi

dataset_sources=(
  "gta:${GTA_ROOT:-${DATA_ROOT:+${DATA_ROOT}/gta}}"
  "synthia:${SYNTHIA_ROOT:-${DATA_ROOT:+${DATA_ROOT}/synthia}}"
  "cityscapes:${CITYSCAPES_ROOT:-${DATA_ROOT:+${DATA_ROOT}/cityscapes}}"
  "acdc:${ACDC_ROOT:-${DATA_ROOT:+${DATA_ROOT}/acdc}}"
  "muses:${MUSES_ROOT:-${DATA_ROOT:+${DATA_ROOT}/muses}}"
  "mapillary:${MAPILLARY_ROOT:-${DATA_ROOT:+${DATA_ROOT}/mapillary}}"
)
for item in "${dataset_sources[@]}"; do
  dataset="${item%%:*}"
  source_dir="${item#*:}"
  link_path="${REPO_ROOT}/data/${dataset}"
  if [[ -z "${source_dir}" || ! -d "${source_dir}" ]]; then
    root_var="$(printf '%s_ROOT' "${dataset^^}")"
    echo "Missing ${dataset} dataset. Set DATA_ROOT or ${root_var}." >&2
    exit 1
  fi
  if [[ -e "${link_path}" && ! -L "${link_path}" ]]; then
    echo "Refusing to replace non-symlink path: ${link_path}" >&2
    exit 1
  fi
  ln -sfn "${source_dir}" "${link_path}"
done

mkdir -p "${REPO_ROOT}/pretrained/dinov3"
resnet_source="${RESNET101_PRETRAINED:-${PRETRAINED_ROOT:+${PRETRAINED_ROOT}/resnet/resnet101-5d3b4d8f.pth}}"
dino_source="${DINOV3_PRETRAINED:-${PRETRAINED_ROOT:+${PRETRAINED_ROOT}/dinov3/dinov3_vitb16.pth}}"
if [[ -z "${resnet_source}" || ! -f "${resnet_source}" ]]; then
  echo "Set PRETRAINED_ROOT or RESNET101_PRETRAINED." >&2
  exit 1
fi
if [[ -L "${REPO_ROOT}/pretrained/resnet101-5d3b4d8f.pth" ||
      ! -f "${REPO_ROOT}/pretrained/resnet101-5d3b4d8f.pth" ]]; then
  cp --remove-destination \
    "${resnet_source}" \
    "${REPO_ROOT}/pretrained/resnet101-5d3b4d8f.pth"
fi

if [[ -L "${REPO_ROOT}/pretrained/dinov3/dinov3_vitb16.pth" ||
      ! -f "${REPO_ROOT}/pretrained/dinov3/dinov3_vitb16.pth" ]]; then
  if [[ -n "${dino_source}" && -f "${dino_source}" ]]; then
    cp --remove-destination \
      "${dino_source}" \
      "${REPO_ROOT}/pretrained/dinov3/dinov3_vitb16.pth"
  elif [[ -n "${DINOV3_ARCHIVE:-}" && -f "${DINOV3_ARCHIVE}" ]]; then
    unzip -j \
      "${DINOV3_ARCHIVE}" \
      dinov3_vitb16.pth \
      -d "${REPO_ROOT}/pretrained/dinov3"
  else
    echo "Set PRETRAINED_ROOT, DINOV3_PRETRAINED, or DINOV3_ARCHIVE." >&2
    exit 1
  fi
fi

for relative in \
  splits/gta/train.txt \
  splits/synthia/train.txt \
  splits/cityscapes/train.txt \
  splits/cityscapes/val.txt \
  splits/acdc/train.txt \
  splits/acdc/val.txt \
  splits/muses/train.txt \
  splits/muses/val.txt \
  splits/mapillary/train.txt \
  splits/mapillary/val.txt; do
  if [[ ! -f "${REPO_ROOT}/${relative}" ]]; then
    echo "Missing dataset list: ${REPO_ROOT}/${relative}" >&2
    exit 1
  fi
done

conda run -n "${CONDA_ENV}" python -c \
  "import sklearn, tensorboardX, timm, torch, torchvision, yaml; print('MADAv2 environment OK:', torch.__version__)"

echo "Environment: ${CONDA_ENV}"
echo "Dataset links: ${REPO_ROOT}/data/{gta,synthia,cityscapes,acdc,muses,mapillary}"
echo "MADAv2 assets are ready."
