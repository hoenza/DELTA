# DELTA

Public-release staging repo for the validated DELTA selective KV-cache path.

## Supported workflows in this repo

Stage 1 supports only the validated core workflows:

- `tests/reason.py` accuracy evaluation on reasoning datasets
- `tests/reasoni.py` baseline-vs-DELTA throughput evaluation on synthetic dummy prompts

The validated default DELTA path is:

- page selector `v2`
- planner dump buffer `fp32`
- page-score implementation `del3_legacy_softmax`
- graph-enabled decode path

## Environment setup

Create the supported env:

```bash
conda env create -f environment.yml
conda activate delta_final
```

Install the validated CUDA/PyTorch/FlashInfer stack:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install flashinfer==0.5.2 -i https://flashinfer.ai/whl/cu121/torch2.5/
```

Install activation hooks for the host compiler and allocator settings:

```bash
bash scripts/install_delta_final_env_hooks.sh
```

## Checkpoints

The repo does not ship checkpoints. Place converted checkpoints under `checkpoints/...` or pass explicit `--model` / `--model_name` arguments.

## Accuracy run

```bash
bash run_reason_accuracy_8gpu.sh
```

## Speed run

```bash
GPU=0 bash run_reasoni_speed_compare.sh
```
