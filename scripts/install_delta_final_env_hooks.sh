#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${1:-${CONDA_PREFIX:-}}
if [[ -z "$ENV_ROOT" ]]; then
    echo "Pass an env root or activate the target environment first." >&2
    exit 1
fi
if [[ ! -d "$ENV_ROOT" ]]; then
    echo "Environment root not found: $ENV_ROOT" >&2
    exit 1
fi

ACTIVATE_DIR="$ENV_ROOT/etc/conda/activate.d"
DEACTIVATE_DIR="$ENV_ROOT/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/delta_toolchain.sh" <<'EOF'
#!/usr/bin/env bash
export DELTA_OLD_CC="${CC-}"
export DELTA_OLD_CXX="${CXX-}"
export DELTA_OLD_CUDAHOSTCXX="${CUDAHOSTCXX-}"
export DELTA_OLD_PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF-}"
export DELTA_OLD_NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export PYTORCH_ALLOC_CONF=expandable_segments:True
EOF

cat > "$DEACTIVATE_DIR/delta_toolchain.sh" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${DELTA_OLD_CC+x}" ]]; then
    if [[ -n "$DELTA_OLD_CC" ]]; then export CC="$DELTA_OLD_CC"; else unset CC; fi
    unset DELTA_OLD_CC
fi
if [[ -n "${DELTA_OLD_CXX+x}" ]]; then
    if [[ -n "$DELTA_OLD_CXX" ]]; then export CXX="$DELTA_OLD_CXX"; else unset CXX; fi
    unset DELTA_OLD_CXX
fi
if [[ -n "${DELTA_OLD_CUDAHOSTCXX+x}" ]]; then
    if [[ -n "$DELTA_OLD_CUDAHOSTCXX" ]]; then export CUDAHOSTCXX="$DELTA_OLD_CUDAHOSTCXX"; else unset CUDAHOSTCXX; fi
    unset DELTA_OLD_CUDAHOSTCXX
fi
if [[ -n "${DELTA_OLD_PYTORCH_ALLOC_CONF+x}" ]]; then
    if [[ -n "$DELTA_OLD_PYTORCH_ALLOC_CONF" ]]; then export PYTORCH_ALLOC_CONF="$DELTA_OLD_PYTORCH_ALLOC_CONF"; else unset PYTORCH_ALLOC_CONF; fi
    unset DELTA_OLD_PYTORCH_ALLOC_CONF
fi
if [[ -n "${DELTA_OLD_NVCC_PREPEND_FLAGS+x}" ]]; then
    if [[ -n "$DELTA_OLD_NVCC_PREPEND_FLAGS" ]]; then export NVCC_PREPEND_FLAGS="$DELTA_OLD_NVCC_PREPEND_FLAGS"; else unset NVCC_PREPEND_FLAGS; fi
    unset DELTA_OLD_NVCC_PREPEND_FLAGS
fi
EOF

chmod +x "$ACTIVATE_DIR/delta_toolchain.sh" "$DEACTIVATE_DIR/delta_toolchain.sh"
echo "Installed activation hooks under $ENV_ROOT"
