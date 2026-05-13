import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize aggregate accuracy JSONs from the overnight reason matrix sweep."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory of the overnight sweep",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def main():
    args = parse_args()
    rows = []

    for aggregate_path in sorted(args.root.rglob("aggregate_*.json")):
        try:
            rel = aggregate_path.relative_to(args.root)
        except ValueError:
            rel = aggregate_path
        parts = rel.parts
        if len(parts) < 4:
            continue

        mode = parts[0]
        env_name = parts[1]
        dataset = parts[2]
        data = load_json(aggregate_path)
        rows.append(
            {
                "mode": mode,
                "env": env_name,
                "dataset": dataset,
                "num_runs": data["num_runs"],
                "mean_accuracy": data["mean_accuracy"],
                "stdev_accuracy": data["stdev_accuracy"],
                "min_accuracy": data["min_accuracy"],
                "max_accuracy": data["max_accuracy"],
                "mean_correct_predictions": data["mean_correct_predictions"],
                "aggregate_json": str(aggregate_path),
            }
        )

    payload = {
        "root": str(args.root),
        "num_aggregate_files": len(rows),
        "rows": rows,
    }

    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote summary JSON to {args.output}")


if __name__ == "__main__":
    main()
