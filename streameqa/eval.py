import argparse
import os

from streameqa.benchmark import InferOnlineBench
from streameqa.data import load_json, save_json
from streameqa.models import build_model


def merge_existing_predictions(data, output_file):
    if not os.path.exists(output_file):
        return data
    existing = load_json(output_file)
    by_id = {item.get("id"): item for item in existing if "id" in item}
    merged = []
    for item in data:
        old = by_id.get(item.get("id"))
        merged.append(old if old is not None else item)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Evaluate StreamEQA with the online benchmark.")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--model_name", default="Qwen3VL")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--context_time", type=float, default=64)
    parser.add_argument(
        "--clip_cache",
        default=None,
        help="Directory for generated clips. By default, use the original cache layout: <video_dir>/tmp_64/.",
    )
    parser.add_argument("--clip_width", type=int, default=360)
    parser.add_argument("--clip_height", type=int, default=420)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = load_json(args.data_file)
    if not args.overwrite:
        data = merge_existing_predictions(data, args.output_file)

    model = build_model(args.model_name, model_path=args.model_path)
    bench = InferOnlineBench(
        video_root=args.video_root,
        clip_cache=args.clip_cache,
        context_time=args.context_time,
        width=args.clip_width,
        height=args.clip_height,
        skip_existing=not args.overwrite,
    )
    summary = bench.eval(data, model, args.output_file)
    save_json(data, args.output_file)
    print(summary)


if __name__ == "__main__":
    main()
