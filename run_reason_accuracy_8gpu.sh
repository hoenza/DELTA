#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$SCRIPT_DIR"
MODEL=${MODEL:-$REPO_ROOT/checkpoints/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/model.pth}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
RUN_NAME=${RUN_NAME:-delta_graph}
ENV_LABEL=${ENV_LABEL:-${CONDA_DEFAULT_ENV:-active}}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
PYTHON_BIN=${PYTHON_BIN:-python}

B=${B:-24}
MAX_LEN=${MAX_LEN:-16384}
NUM_SAMPLES=${NUM_SAMPLES:-30}
FULL_CACHE_LAYERS=(${FULL_CACHE_LAYERS:-0 1 2 14 22})
SUBSET_CACHE_SIZE=${SUBSET_CACHE_SIZE:-64}
COMPRESSION_RATIO=${COMPRESSION_RATIO:-1}
L=${L:-8}
PRINT_OUTPUT=${PRINT_OUTPUT:-0}
DATASETS=(${DATASETS:-aime2024 aime2025})
GPU_IDS=(${GPU_IDS:-0 1 2 3 4 5 6 7})

CC_BIN=${CC_BIN:-/usr/bin/gcc}
CXX_BIN=${CXX_BIN:-/usr/bin/g++}
CUDAHOSTCXX_BIN=${CUDAHOSTCXX_BIN:-/usr/bin/g++}
NVCC_HOST_COMPILER_BIN=${NVCC_HOST_COMPILER_BIN:-/usr/bin/g++-11}
PYTORCH_ALLOC_CONF_VALUE=${PYTORCH_ALLOC_CONF_VALUE:-expandable_segments:True}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs/reason_accuracy_8gpu_$(date +%Y%m%d_%H%M%S)}

AGGREGATE_SCRIPT=${AGGREGATE_SCRIPT:-$REPO_ROOT/tests/aggregate_reason_accuracies.py}
SUMMARY_SCRIPT=${SUMMARY_SCRIPT:-$REPO_ROOT/tests/summarize_reason_matrix.py}

mkdir -p "$LOG_DIR"

if [[ ! -f "$MODEL" ]]; then
    echo "Model checkpoint not found: $MODEL" >&2
    exit 1
fi

print_command() { printf '%q ' "$@"; printf '
'; }
extract_results_json() { grep "All results saved to:" "$1" | tail -1 | sed 's/.*All results saved to:[[:space:]]*//' | tr -d ''; }

run_single() {
    local dataset="$1"
    local gpu="$2"
    local run_root="$LOG_DIR/$RUN_NAME/$ENV_LABEL/$dataset/gpu${gpu}"
    local run_dir="$run_root/cwd"
    local logfile="$run_root/run.log"
    local cache_root="$run_root/cache"
    mkdir -p "$run_dir" "$cache_root/home" "$cache_root/xdg" "$cache_root/flashinfer"

    local -a cmd=(
        env
        "PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        "CUDA_VISIBLE_DEVICES=$gpu"
        "ENABLE_INTRA_NODE_COMM=1"
        "CC=$CC_BIN"
        "CXX=$CXX_BIN"
        "CUDAHOSTCXX=$CUDAHOSTCXX_BIN"
        "NVCC_PREPEND_FLAGS=-ccbin $NVCC_HOST_COMPILER_BIN"
        "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF_VALUE"
        "HOME=$cache_root/home"
        "XDG_CACHE_HOME=$cache_root/xdg"
        "FLASHINFER_WORKSPACE_BASE=$cache_root/flashinfer"
        "$TORCHRUN_BIN"
        --standalone
        --nproc_per_node=1
        "$REPO_ROOT/tests/reason.py"
        --model "$MODEL"
        --model_name "$MODEL_NAME"
        --rank_group 0
        --B "$B"
        --max_len "$MAX_LEN"
        --dataset "$dataset"
        --num_samples "$NUM_SAMPLES"
        --cuda_graph_decode
        --enable_selective_cache
        --full_cache_layers "${FULL_CACHE_LAYERS[@]}"
        --subset_cache_size "$SUBSET_CACHE_SIZE"
        --compression_ratio "$COMPRESSION_RATIO"
        --L "$L"
    )
    if [[ "$PRINT_OUTPUT" == "1" ]]; then cmd+=(--printoutput); fi

    {
        echo
        echo "============================================================"
        echo "  Run: $RUN_NAME"
        echo "  Env: $ENV_LABEL"
        echo "  Dataset: $dataset"
        echo "  GPU: $gpu"
        echo "  Root: $REPO_ROOT"
        echo "  Log: $logfile"
        echo "============================================================"
        echo "Command:"
        print_command "${cmd[@]}"
    } > "$logfile"

    (
        cd "$run_dir"
        "${cmd[@]}"
    ) >> "$logfile" 2>&1
}

collect_result_path() {
    local dataset="$1"
    local gpu="$2"
    local logfile="$LOG_DIR/$RUN_NAME/$ENV_LABEL/$dataset/gpu${gpu}/run.log"
    local run_dir="$LOG_DIR/$RUN_NAME/$ENV_LABEL/$dataset/gpu${gpu}/cwd"
    local result_ref
    result_ref="$(extract_results_json "$logfile")"
    if [[ -z "$result_ref" ]]; then
        echo "Could not find result JSON in $logfile" >&2
        return 1
    fi
    if [[ "$result_ref" = /* ]]; then printf '%s
' "$result_ref"; else printf '%s
' "$run_dir/$result_ref"; fi
}

run_batch() {
    local dataset="$1"
    local batch_root="$LOG_DIR/$RUN_NAME/$ENV_LABEL/$dataset"
    mkdir -p "$batch_root"
    echo
    echo "Launching ${#GPU_IDS[@]} parallel runs:"
    echo "  run=$RUN_NAME env=$ENV_LABEL dataset=$dataset"

    local pids=() failed=0
    for gpu in "${GPU_IDS[@]}"; do run_single "$dataset" "$gpu" & pids+=("$!"); done
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    if [[ "$failed" -ne 0 ]]; then
        echo "At least one run failed for run=$RUN_NAME env=$ENV_LABEL dataset=$dataset" >&2
        return 1
    fi

    local result_paths=()
    for gpu in "${GPU_IDS[@]}"; do result_paths+=("$(collect_result_path "$dataset" "$gpu")"); done
    "$PYTHON_BIN" "$AGGREGATE_SCRIPT" --dataset "$dataset" --inputs "${result_paths[@]}" --output "$batch_root/aggregate_${dataset}.json" > "$batch_root/aggregate_stdout.log" 2>&1
}

status_tsv="$LOG_DIR/status.tsv"
printf 'run	env	dataset	status
' > "$status_tsv"

for dataset in "${DATASETS[@]}"; do
    if run_batch "$dataset"; then
        printf '%s	%s	%s	%s
' "$RUN_NAME" "$ENV_LABEL" "$dataset" ok >> "$status_tsv"
    else
        printf '%s	%s	%s	%s
' "$RUN_NAME" "$ENV_LABEL" "$dataset" failed >> "$status_tsv"
        echo "Continuing after failure: run=$RUN_NAME env=$ENV_LABEL dataset=$dataset" | tee -a "$LOG_DIR/failures.log"
    fi
done

"$PYTHON_BIN" "$SUMMARY_SCRIPT" --root "$LOG_DIR" --output "$LOG_DIR/summary.json" > "$LOG_DIR/summary_stdout.log" 2>&1 || true

echo
echo "DELTA accuracy sweep completed."
echo "Logs are under: $LOG_DIR"
echo "Status table: $status_tsv"
