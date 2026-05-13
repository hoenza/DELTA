#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
CC_BIN=${CC_BIN:-/usr/bin/gcc}
CXX_BIN=${CXX_BIN:-/usr/bin/g++}
CUDAHOSTCXX_BIN=${CUDAHOSTCXX_BIN:-/usr/bin/g++}
NVCC_HOST_COMPILER_BIN=${NVCC_HOST_COMPILER_BIN:-/usr/bin/g++-11}
PYTORCH_ALLOC_CONF_VALUE=${PYTORCH_ALLOC_CONF_VALUE:-expandable_segments:True}

MODEL=${MODEL:-checkpoints/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/model.pth}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
REASONI=${REASONI:-tests/reasoni.py}

B=${B:-64}
MAX_LEN=${MAX_LEN:-18432}
DATASET=${DATASET:-dummy4tok}
NUM_SAMPLES=${NUM_SAMPLES:-64}
SMOOTH=${SMOOTH:-50}
GPU=${GPU:-0}
COMPRESSION_RATIO=${COMPRESSION_RATIO:-1}
SUBSET_CACHE_SIZE=${SUBSET_CACHE_SIZE:-64}
FULL_CACHE_LAYERS=(${FULL_CACHE_LAYERS:-0 1 2 14 22})
DUMMY_TOKEN_ID=${DUMMY_TOKEN_ID:-}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/plots/fig4/logs/reasoni_speed_compare_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$LOG_DIR"

extract_json() {
    local logfile="$1"
    local jsonpath
    jsonpath=$(grep "Step timings saved to:" "$logfile" | tail -1 | sed 's/.*Step timings saved to:[[:space:]]*//' | tr -d '')
    if [[ -z "$jsonpath" ]]; then echo "ERROR: No 'Step timings saved to:' line found in $logfile" >&2; return 1; fi
    if [[ ! -f "$jsonpath" ]]; then echo "ERROR: Timing JSON referenced by $logfile does not exist: $jsonpath" >&2; return 1; fi
    printf '%s
' "$jsonpath"
}

copy_json_for_traceability() { cp "$1" "$2"; printf '%s
' "$2"; }

BASE_CMD=(
    "$TORCHRUN_BIN"
    --standalone
    --nproc_per_node=1
    "$REASONI"
    --model "$MODEL"
    --model_name "$MODEL_NAME"
    --rank_group 0
    --B "$B"
    --max_len "$MAX_LEN"
    --dataset "$DATASET"
    --num_samples "$NUM_SAMPLES"
    --cuda_graph_decode
)
if [[ -n "$DUMMY_TOKEN_ID" ]]; then BASE_CMD+=(--dummy_token_id "$DUMMY_TOKEN_ID"); fi

DELTA_ARGS=(
    --enable_selective_cache
    --full_cache_layers "${FULL_CACHE_LAYERS[@]}"
    --subset_cache_size "$SUBSET_CACHE_SIZE"
    --compression_ratio "$COMPRESSION_RATIO"
)

run_case() {
    local label="$1"
    local gpu="$2"
    local logfile="$3"
    shift 3
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export ENABLE_INTRA_NODE_COMM=1
        export CC="$CC_BIN"
        export CXX="$CXX_BIN"
        export CUDAHOSTCXX="$CUDAHOSTCXX_BIN"
        export NVCC_PREPEND_FLAGS="-ccbin $NVCC_HOST_COMPILER_BIN"
        export PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF_VALUE"
        export HOME=/tmp
        export XDG_CACHE_HOME=/tmp/.cache
        export FLASHINFER_WORKSPACE_BASE=/tmp
        "${BASE_CMD[@]}" "$@"
    ) 2>&1 | tee "$logfile"
}

echo "============================================================"
echo "  DELTA reasoni speed compare"
echo "  B=$B  MaxLen=$MAX_LEN  Dataset=$DATASET  Samples=$NUM_SAMPLES"
echo "  CompressionRatio=$COMPRESSION_RATIO  SubsetCacheSize=$SUBSET_CACHE_SIZE"
echo "  FullCacheLayers=${FULL_CACHE_LAYERS[*]}"
echo "  GPU=$GPU"
echo "  Logs: $LOG_DIR"
echo "============================================================"

echo
echo "=== Run 1/2: Baseline (CUDA graph) on GPU $GPU ==="
run_case baseline "$GPU" "$LOG_DIR/1_baseline.log"

echo
echo "=== Run 2/2: DELTA (CUDA graph) on GPU $GPU ==="
run_case delta "$GPU" "$LOG_DIR/2_delta.log" "${DELTA_ARGS[@]}"

BASELINE_JSON=$(extract_json "$LOG_DIR/1_baseline.log")
DELTA_JSON=$(extract_json "$LOG_DIR/2_delta.log")
BASELINE_JSON_COPY=$(copy_json_for_traceability "$BASELINE_JSON" "$LOG_DIR/1_baseline_step_timings.json")
DELTA_JSON_COPY=$(copy_json_for_traceability "$DELTA_JSON" "$LOG_DIR/2_delta_step_timings.json")

OUTPUT_STEM="$LOG_DIR/reasoni_speed_compare"
"$PYTHON_BIN" "$REPO_ROOT/plots/fig4/plot.py"     --baseline-json "$BASELINE_JSON_COPY"     --delta-json "$DELTA_JSON_COPY"     --output-stem "$OUTPUT_STEM"     --smooth "$SMOOTH"

echo
echo "Finished."
echo "Baseline log: $LOG_DIR/1_baseline.log"
echo "DELTA log: $LOG_DIR/2_delta.log"
echo "Baseline JSON: $BASELINE_JSON_COPY"
echo "DELTA JSON: $DELTA_JSON_COPY"
echo "Model-forward PDF: $OUTPUT_STEM"_model_forward.pdf
echo "Model-forward PNG: $OUTPUT_STEM"_model_forward.png
echo "Breakdown PDF: $OUTPUT_STEM"_breakdown.pdf
echo "Breakdown PNG: $OUTPUT_STEM"_breakdown.png
