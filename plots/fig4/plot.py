from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COLORS = SCRIPT_DIR.parent / "fig3" / "colors.json"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "qwen15b_timing_compare"
DEFAULT_BASELINE_JSON = SCRIPT_DIR / "baseline_step_timings.json"
DEFAULT_DELTA_JSON = SCRIPT_DIR / "delta64_step_timings.json"
DEFAULT_SMOOTH = 50
DEFAULT_SKIP_FIRST = 100
SPIKE_WINDOW = 5
SPIKE_RATIO = 1.35
SPIKE_ABS_MS = 3.0
SPIKE_MAX_RUN = 3
COMPONENT_ORDER = ["collect_pages", "plan", "model_forward", "postprocess"]
COMPONENT_COLORS = {
    "collect_pages": "#6c59c2",
    "plan": "#e98020",
    "model_forward": "#600aa6",
    "postprocess": "#f9b54f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot baseline vs DELTA timing figures across decoding rounds."
    )
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--delta-json", type=Path, default=DEFAULT_DELTA_JSON)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help=(
            "Output path without extension. "
            f"Defaults to {DEFAULT_OUTPUT_STEM} and writes PDF and PNG."
        ),
    )
    parser.add_argument(
        "--colors",
        type=Path,
        default=DEFAULT_COLORS,
        help=f"Color palette JSON path (default: {DEFAULT_COLORS})",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=DEFAULT_SMOOTH,
        help=f"Moving-average window size (default: {DEFAULT_SMOOTH}).",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=DEFAULT_SKIP_FIRST,
        help=f"Skip this many initial decode rounds before plotting (default: {DEFAULT_SKIP_FIRST}).",
    )
    return parser.parse_args()


def load_colors(path: Path) -> dict[str, str]:
    with path.open() as f:
        payload = json.load(f)

    palette = payload.get("palette", [])
    if len(palette) < 3:
        raise ValueError("Palette must contain at least 3 colors.")
    return {
        "Full": str(palette[2]).strip(),
        "DELTA": str(palette[0]).strip(),
    }


def load_timings(path: Path) -> tuple[dict, list[dict]]:
    with path.open() as f:
        data = json.load(f)
    if "meta" not in data or "steps" not in data:
        raise ValueError(f"{path} does not look like a step timings JSON file.")
    if not isinstance(data["steps"], list) or not data["steps"]:
        raise ValueError(f"{path} does not contain any timing steps.")
    return data["meta"], data["steps"]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if len(values) < window:
        raise ValueError(
            f"Cannot smooth {len(values)} points with window {window}; reduce --smooth."
        )
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def drop_initial_steps(steps: list[dict], skip_first: int) -> list[dict]:
    if skip_first <= 0:
        return steps
    if len(steps) <= skip_first:
        raise ValueError(
            f"Cannot skip {skip_first} steps when only {len(steps)} timing steps are available."
        )
    return steps[skip_first:]


def drop_terminal_step(steps: list[dict]) -> list[dict]:
    if len(steps) <= 1:
        raise ValueError("Need at least 2 timing steps to drop the final decode round.")
    return steps[:-1]


def suppress_transient_spikes(
    values: np.ndarray,
    window: int = SPIKE_WINDOW,
    ratio: float = SPIKE_RATIO,
    abs_ms: float = SPIKE_ABS_MS,
    max_run: int = SPIKE_MAX_RUN,
) -> np.ndarray:
    if len(values) < (2 * window + 1):
        return values

    filtered = values.copy()
    spike_mask = np.zeros(len(values), dtype=bool)
    for idx in range(window, len(values) - window):
        neighbors = np.concatenate((values[idx - window : idx], values[idx + 1 : idx + window + 1]))
        local_median = float(np.median(neighbors))
        if values[idx] > max(local_median * ratio, local_median + abs_ms):
            spike_mask[idx] = True

    run_start = None
    for idx in range(len(values) + 1):
        is_spike = idx < len(values) and spike_mask[idx]
        if is_spike and run_start is None:
            run_start = idx
        elif not is_spike and run_start is not None:
            run_end = idx
            if run_end - run_start <= max_run:
                left = max(0, run_start - window)
                right = min(len(values), run_end + window)
                neighbors = np.concatenate((values[left:run_start], values[run_end:right]))
                if len(neighbors):
                    filtered[run_start:run_end] = np.median(neighbors)
            run_start = None

    return filtered


def smoothed_xy(steps: list[dict], key: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    rounds = np.asarray([float(step["step"]) for step in steps], dtype=float)
    values = np.asarray([float(step[key]) for step in steps], dtype=float)
    if key == "model_forward_ms":
        values = suppress_transient_spikes(values)
    if window <= 1:
        return rounds, values
    return moving_average(rounds, window), moving_average(values, window)


def smoothed_components(
    steps: list[dict], window: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rounds = np.asarray([float(step["step"]) for step in steps], dtype=float)
    series: dict[str, np.ndarray] = {}
    for component in COMPONENT_ORDER:
        values = np.asarray([float(step[f"{component}_ms"]) for step in steps], dtype=float)
        if component == "model_forward":
            values = suppress_transient_spikes(values)
        series[component] = moving_average(values, window) if window > 1 else values
    if window > 1:
        rounds = moving_average(rounds, window)
    return rounds, series


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 12


def dataset_label(meta: dict) -> str:
    dataset = str(meta.get("dataset", "AIME")).strip()
    return dataset.upper() if dataset.lower() == "aime" else dataset


def model_label(meta: dict) -> str:
    model_name = str(meta.get("model_name", "Qwen-1.5B")).strip()
    if model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        return "Qwen-1.5B"
    return model_name.split("/")[-1]


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")


def build_model_forward_plot(
    baseline_meta: dict,
    baseline_steps: list[dict],
    delta_meta: dict,
    delta_steps: list[dict],
    colors: dict[str, str],
    output_stem: Path,
    smooth: int,
    skip_first: int,
) -> None:
    baseline_plot_steps = drop_terminal_step(drop_initial_steps(baseline_steps, skip_first))
    delta_plot_steps = drop_terminal_step(drop_initial_steps(delta_steps, skip_first))
    baseline_x, baseline_y = smoothed_xy(baseline_plot_steps, "model_forward_ms", smooth)
    delta_x, delta_y = smoothed_xy(delta_plot_steps, "model_forward_ms", smooth)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(baseline_x, baseline_y, linewidth=2.2, color=colors["Full"], label="Full")
    ax.plot(delta_x, delta_y, linewidth=2.2, color=colors["DELTA"], label="DELTA")
    ax.set_xlabel("Decoding Round")
    ax.set_ylabel("Model Forward Runtime (ms)")
    ax.set_title(model_label(delta_meta))
    ax.grid(True, linewidth=0.6, alpha=0.7)

    handles = [
        Line2D([0], [0], linewidth=2.2, color=colors["Full"], label="Full"),
        Line2D([0], [0], linewidth=2.2, color=colors["DELTA"], label="DELTA"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=16,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#bdbdbd",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, output_stem)


def build_breakdown_plot(
    baseline_meta: dict,
    baseline_steps: list[dict],
    delta_meta: dict,
    delta_steps: list[dict],
    output_stem: Path,
    smooth: int,
    skip_first: int,
) -> None:
    baseline_plot_steps = drop_terminal_step(drop_initial_steps(baseline_steps, skip_first))
    delta_plot_steps = drop_terminal_step(drop_initial_steps(delta_steps, skip_first))
    baseline_x, baseline_components = smoothed_components(baseline_plot_steps, smooth)
    delta_x, delta_components = smoothed_components(delta_plot_steps, smooth)
    baseline_total = sum(baseline_components[component] for component in COMPONENT_ORDER)
    delta_total = sum(delta_components[component] for component in COMPONENT_ORDER)
    common_ylim = 1.05 * max(float(np.max(baseline_total)), float(np.max(delta_total)))

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.75), sharex="col", squeeze=False)
    specs = [
        ("Full", baseline_meta, baseline_x, baseline_components, axes[0][0], axes[1][0]),
        ("DELTA", delta_meta, delta_x, delta_components, axes[0][1], axes[1][1]),
    ]

    tick_formatter = FuncFormatter(
        lambda x, _pos: f"{int(x / 1000)}k" if x >= 1000 and x % 1000 == 0 else f"{int(x)}"
    )

    for idx, (label, meta, rounds, components, line_ax, stack_ax) in enumerate(specs):
        total = np.zeros_like(rounds)
        for component in COMPONENT_ORDER:
            total = total + components[component]
            line_ax.plot(
                rounds,
                components[component],
                linewidth=1.8,
                color=COMPONENT_COLORS[component],
                label=component.replace("_", " "),
            )
        line_ax.plot(rounds, total, linewidth=2.0, color="black", label="step total")
        line_ax.set_title(label)
        line_ax.grid(True, linewidth=0.6, alpha=0.7)
        line_ax.set_ylim(0, common_ylim)
        if idx == 0:
            line_ax.set_ylabel("Runtime (ms)", fontsize=16)

        bottom = np.zeros_like(rounds)
        for component in COMPONENT_ORDER:
            top = bottom + components[component]
            stack_ax.fill_between(
                rounds,
                bottom,
                top,
                color=COMPONENT_COLORS[component],
                alpha=0.8,
                label=component.replace("_", " "),
            )
            stack_ax.plot(rounds, top, linewidth=1.0, color=COMPONENT_COLORS[component], alpha=0.95)
            bottom = top
        stack_ax.grid(True, linewidth=0.6, alpha=0.7)
        stack_ax.set_ylim(0, common_ylim)
        stack_ax.set_xlabel("Decoding Round", fontsize=20)
        stack_ax.xaxis.set_major_locator(MultipleLocator(5000))
        stack_ax.xaxis.set_major_formatter(tick_formatter)
        if idx == 0:
            stack_ax.set_ylabel("Runtime (ms)", fontsize=16)

    handles = [
        Line2D([0], [0], linewidth=8, color=COMPONENT_COLORS[component], label=component.replace("_", " "))
        for component in COMPONENT_ORDER
    ] + [Line2D([0], [0], linewidth=2.0, color="black", label="step total")]
    fig.suptitle(
        f"DS-{model_label(delta_meta)}",
        y=0.95,
        fontsize=24,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=16,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#bdbdbd",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.945], h_pad=1.0, w_pad=1.5)
    save_figure(fig, output_stem)


def main() -> None:
    args = parse_args()
    apply_style()
    colors = load_colors(args.colors)
    baseline_meta, baseline_steps = load_timings(args.baseline_json)
    delta_meta, delta_steps = load_timings(args.delta_json)

    build_model_forward_plot(
        baseline_meta=baseline_meta,
        baseline_steps=baseline_steps,
        delta_meta=delta_meta,
        delta_steps=delta_steps,
        colors=colors,
        output_stem=args.output_stem.with_name(f"{args.output_stem.name}_model_forward"),
        smooth=args.smooth,
        skip_first=args.skip_first,
    )
    build_breakdown_plot(
        baseline_meta=baseline_meta,
        baseline_steps=baseline_steps,
        delta_meta=delta_meta,
        delta_steps=delta_steps,
        output_stem=args.output_stem.with_name(f"{args.output_stem.name}_breakdown"),
        smooth=args.smooth,
        skip_first=args.skip_first,
    )


if __name__ == "__main__":
    main()
