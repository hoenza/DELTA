#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME=${ENV_NAME:-delta_final}
CONDA_SH=${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Could not locate conda.sh. Set CONDA_SH explicitly." >&2
    exit 1
fi

source "$CONDA_SH"
conda env create -n "$ENV_NAME" -f "$REPO_ROOT/environment.yml"
conda activate "$ENV_NAME"
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install flashinfer==0.5.2 -i https://flashinfer.ai/whl/cu121/torch2.5/
bash "$REPO_ROOT/scripts/install_delta_final_env_hooks.sh"

echo "delta_final environment is ready."
