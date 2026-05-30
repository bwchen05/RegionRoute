#!/usr/bin/env bash

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="flux_env/bin/activate"

STYLE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --style)
            STYLE="${2:-}"
            shift 2
            ;;
        -h|--help)
            echo "usage: train.sh --style <name>"
            echo "env overrides: DATA_PARQUET, JSON_DIR, JSON_PATH, DATASET_PATH, OUTPUT_DIR, LORA_DEST_DIR"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$STYLE" ]]; then
    echo "error: --style is required" >&2
    echo "usage: train.sh --style <name>" >&2
    exit 1
fi

if [[ -f "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
else
    echo "warning: venv not found at $VENV_ACTIVATE -- using system python" >&2
fi

DATA_PARQUET="${DATA_PARQUET:-${REPO_DIR}/style.parquet}"
JSON_DIR="${JSON_DIR:-${REPO_DIR}/prompts}"
JSON_PATH="${JSON_PATH:-${JSON_DIR}/${STYLE}.json}"
DATASET_PATH="${DATASET_PATH:-./dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/runs/${STYLE}}"
LORA_DEST_DIR="${LORA_DEST_DIR:-${REPO_DIR}/lora_out}"

echo "[train.sh] style=${STYLE}"
echo "[train.sh] json_path=${JSON_PATH}"
echo "[train.sh] output_dir=${OUTPUT_DIR}"
echo "[train.sh] final lora destination=${LORA_DEST_DIR}/${STYLE}.safetensors"

accelerate launch --config_file="${REPO_DIR}/accelerate.yaml" \
    "${REPO_DIR}/lora.py" \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev" \
    --data_df_path="${DATA_PARQUET}" \
    --json_path="${JSON_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --mixed_precision="bf16" \
    --use_8bit_adam \
    --weighting_scheme="none" \
    --resolution=1024 \
    --train_batch_size=2 \
    --repeats=1 \
    --learning_rate=1e-5 \
    --focus_loss_scale=0.1 \
    --cover_loss_scale=5e-5 \
    --guidance_scale=1 \
    --gradient_accumulation_steps=4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --cache_latents \
    --rank=4 \
    --max_train_steps=10000 \
    --seed=0 \
    --gradient_checkpointing \
    --report_to="wandb"

mkdir -p "${LORA_DEST_DIR}"
SRC_WEIGHTS="${OUTPUT_DIR}/pytorch_lora_weights.safetensors"
DEST_WEIGHTS="${LORA_DEST_DIR}/${STYLE}.safetensors"
if [[ -f "${SRC_WEIGHTS}" ]]; then
    cp "${SRC_WEIGHTS}" "${DEST_WEIGHTS}"
    echo "[train.sh] copied ${SRC_WEIGHTS} -> ${DEST_WEIGHTS}"
else
    echo "[train.sh] warning: ${SRC_WEIGHTS} not found; nothing to copy" >&2
fi
