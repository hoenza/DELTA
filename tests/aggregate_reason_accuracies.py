import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate accuracy across multiple reason.py result JSONs."
    )
    parser.add_argument("--dataset", required=True, help="Dataset label for reporting")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Result JSON paths to aggregate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


def load_result(path: Path):
    with path.open() as f:
        return json.load(f)


def main():
    args = parse_args()

    rows = []
    for path in args.inputs:
        data = load_result(path)
        summary = data.get("summary", {})
        rows.append(
            {
                "path": str(path),
                "accuracy": float(summary["accuracy"]),
                "correct_predictions": int(summary["correct_predictions"]),
                "total_samples": int(summary["total_samples"]),
            }
        )

    accuracies = [row["accuracy"] for row in rows]
    correct_counts = [row["correct_predictions"] for row in rows]
    total_samples_set = sorted({row["total_samples"] for row in rows})

    payload = {
        "dataset": args.dataset,
        "num_runs": len(rows),
        "total_samples_values": total_samples_set,
        "mean_accuracy": statistics.fmean(accuracies),
        "stdev_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "mean_correct_predictions": statistics.fmean(correct_counts),
        "runs": rows,
    }

    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote aggregate JSON to {args.output}")


if __name__ == "__main__":
    main()
