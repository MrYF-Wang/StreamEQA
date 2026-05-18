import argparse
import csv
from collections import defaultdict
from pathlib import Path

from streameqa.data import load_json, save_json


GT_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def compute_stats(data, model_name):
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for item in data:
        if model_name not in item or item.get(model_name) in (None, ""):
            continue
        pred = str(item.get(model_name))[0]
        gt = GT_TO_LETTER.get(item.get("gt"))
        task = item.get("task")
        stats[task]["total"] += 1
        stats["total"]["total"] += 1
        if pred == gt:
            stats[task]["correct"] += 1
            stats["total"]["correct"] += 1
    for values in stats.values():
        total = values["total"]
        values["accuracy"] = values["correct"] / total if total else 0.0
    return dict(stats)


def write_csv(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "total", "correct", "accuracy"])
        writer.writeheader()
        for task, values in stats.items():
            writer.writerow({"task_type": task, **values})


def main():
    parser = argparse.ArgumentParser(description="Compute StreamEQA accuracy.")
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    stats = compute_stats(load_json(args.pred_file), args.model_name)
    if args.output_json:
        save_json(stats, args.output_json)
    if args.output_csv:
        write_csv(stats, args.output_csv)
    print(stats.get("total", {"total": 0, "correct": 0, "accuracy": 0.0}))


if __name__ == "__main__":
    main()
