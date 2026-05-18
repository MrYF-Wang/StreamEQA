# [CVPR 2026 Findings] StreamEQA: Towards Streaming Video Understanding for Embodied Scenarios

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Accepted to appear in CVPR 2026 Findings!

This repository contains the official implementation of our paper **StreamEQA: Towards Streaming Video Understanding for Embodied Scenarios**.

## 📄 Abstract

As embodied intelligence advances toward real-world deployment, the ability to continuously perceive and reason over streaming visual inputs becomes essential. In such settings, an agent must maintain situational awareness of its environment, comprehend the interactions with surrounding entities, and dynamically plan actions informed by past observations, current contexts, and anticipated future events. To facilitate progress in this direction, we introduce StreamEQA, the first benchmark designed for streaming video question answering in embodied scenarios. StreamEQA evaluates existing MLLMs along two orthogonal dimensions: Embodied and Streaming. Along the embodied dimension, we categorize the questions into three levels: perception, interaction, and planning, which progressively assess a model's ability to recognize fine-grained visual details, reason about agent-object interactions, and perform high-level goal-directed reasoning. For the streaming dimension, questions are divided into backward, real-time, and forward reasoning, with each mode relying on a distinct temporal context. Built upon 156 independent long videos, StreamEQA defines 42 tasks and generates approximately 21K question-answer pairs with precise timestamps through a hybrid pipeline combining automated generation and human refinement. Evaluations of 13 state-of-the-art video-LLMs reveal that, despite strong performance on conventional benchmarks, these models still struggle with streaming video understanding in embodied scenarios. We hope StreamEQA will catalyze research on streaming video understanding for embodied applications.

## Usage

Run Qwen3VL evaluation from the repository root:

```bash
cd ./StreamEQA
bash scripts/eval_online.sh
```

By default, this evaluates Qwen3VL on `Perception` and writes:

```text
results/Perception_output_Qwen3VL.json
```

Evaluate another annotation split by overriding `TOPIC`:

```bash
TOPIC=Interaction bash scripts/eval_online.sh
TOPIC=Planning_all bash scripts/eval_online.sh
```

You can also invoke the package entry point directly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m streameqa.eval \
  --model_name Qwen3VL \
  --model_path /path/to/Qwen3-VL-8B-Instruct \
  --video_root /path/to/videos \
  --data_file data/annotations/Perception.json \
  --output_file results/Perception_output_Qwen3VL.json
```

Useful options:

```bash
python -m streameqa.eval \
  --model_name Qwen3VL \
  --model_path /path/to/Qwen3-VL-8B-Instruct \
  --video_root /path/to/videos \
  --data_file data/annotations/Planning_all.json \
  --output_file results/Planning_all_output_Qwen3VL.json \
  --context_time 128 \
  --overwrite
```

To pass additional CLI options through the shell script, use `EXTRA_ARGS`:

```bash
EXTRA_ARGS="--context_time 128 --overwrite" bash scripts/eval_online.sh
```

`context_time` is intentionally not set in `eval_online.sh`; the Python CLI default is used unless you explicitly pass `--context_time`.

Compute accuracy:

```bash
python -m streameqa.metrics \
  --pred_file results/Perception_output_Qwen3VL.json \
  --model_name Qwen3VL \
  --output_json results/Perception_Qwen3VL_stats.json \
  --output_csv results/Perception_Qwen3VL_stats.csv
```

## Configuration

Runtime paths are provided through command-line arguments or environment variables in `scripts/eval_online.sh`.

### Expected File Layout

Annotations are included in this repository:

```text
data/
└── annotations/
    ├── Perception.json
    ├── Interaction.json
    └── Planning_all.json
```

Prepare videos locally using this layout:

```text
videos/
├── P01/
│   ├── P01-20240202-171220.mp4
│   └── tmp_64/
├── P02/
├── P03/
├── P08/
└── P09/
```

For an annotation with:

```json
{"video": "P09-20240622-150155"}
```

the evaluator expects:

```text
{VIDEO_ROOT}/P09/P09-20240622-150155.mp4
```

Generated video clips follow the original experimental cache layout:

```text
{VIDEO_ROOT}/{participant}/tmp_64/{video_name}_{int(start)}_{end}.mp4
```

Prepare model checkpoints locally, for example:

```text
models/
└── Qwen3-VL-8B-Instruct/
```

Pass this directory through `--model_path` or `MODEL_PATH`.

### CLI Options

| Option | Purpose | Default |
| --- | --- | --- |
| `--data_file` | Annotation JSON file. | required |
| `--output_file` | Output JSON file storing full QA items plus model predictions. | required |
| `--video_root` | Root directory containing participant video folders such as `P01/`, `P02/`. | required |
| `--model_name` | Model adapter name. | `Qwen3VL` |
| `--model_path` | Local checkpoint directory for the selected model. | unset |
| `--context_time` | Online context window in seconds before query time. | `64` |
| `--clip_cache` | Optional override for generated clip directory. If unset, use `{video_dir}/tmp_64/`. | unset |
| `--clip_width` | Clip resize width. | `360` |
| `--clip_height` | Clip resize height. | `420` |
| `--overwrite` | Ignore existing model predictions in the output file. | disabled |

### Shell Script Defaults

`scripts/eval_online.sh` is configured for the local Qwen3VL setup:

| Variable | Purpose | Default |
| --- | --- | --- |
| `EVAL_MODEL` | Model adapter name. | `Qwen3VL` |
| `TOPIC` | Annotation topic. | `Perception` |
| `DATA_FILE` | Annotation path. | `data/annotations/${TOPIC}.json` |
| `OUTPUT_FILE` | Prediction output path. | `results/${TOPIC}_output_${EVAL_MODEL}.json` |
| `VIDEO_ROOT` | Local video root. | `/path/to/videos` |
| `MODEL_PATH` | Local Qwen3VL checkpoint. | `/path/to/Qwen3-VL-8B-Instruct` |
| `GPU` | CUDA device id. | `0` |

Supported model adapter names:

```text
Qwen3VL, InternVL3-8B, LongVA, MiniCPM-V, VideoLLaMA3, EgoGPT,
EgoVLPv2, VideollmOnline, FlashVstream, Dispider, TimeChatOnline
```

Except for `Qwen3VL`, these adapters preserve the original wrapper logic and path handling from the experimental code. Prepare the corresponding third-party packages, source paths, and checkpoints exactly as required by each model wrapper before running them.

## Code Structure

```text
StreamEQA/
├── scripts/
│   └── eval_online.sh              # Convenience launcher for Qwen3VL online evaluation
├── streameqa/
│   ├── eval.py                     # CLI entry point for online evaluation
│   ├── metrics.py                  # Accuracy computation and CSV/JSON statistics
│   ├── data.py                     # JSON loading, saving, and task filtering helpers
│   ├── prompts.py                  # Multiple-choice prompt formatting and answer parsing
│   ├── benchmark/
│   │   └── infer_online.py         # InferOnline benchmark loop
│   ├── models/
│   │   ├── base.py                 # Minimal model adapter interface
│   │   ├── qwen3vl.py              # Qwen3VL adapter
│   │   └── *.py                    # Legacy model wrappers migrated from the experiments
│   └── utils/
│       └── video.py                # Video path resolution, online windowing, and ffmpeg clipping
├── data/
│   └── annotations/                # Released StreamEQA QA annotations
├── requirements.txt
└── README.md
```

At a high level, `streameqa.eval` loads annotations, instantiates a model adapter, and calls `streameqa.benchmark.infer_online`. The benchmark resolves each source video from `VIDEO_ROOT`, clips the online context window before the query timestamp, formats the multiple-choice prompt, runs the model, and writes the full annotation list with model predictions. `streameqa.metrics` computes per-task and overall accuracy from that output.

## Data Statistics

| File | Level | QA pairs | Task codes |
| --- | ---: | ---: | ---: |
| `data/annotations/Perception.json` | Perception | 6,500 | 13 |
| `data/annotations/Interaction.json` | Interaction | 7,996 | 16 |
| `data/annotations/Planning_all.json` | Planning | 6,235 | 13 |
| Total | StreamEQA | 20,731 | 42 tasks |

## Requirements

- Python 3.10 or newer
- CUDA-capable GPU recommended for local video-LLMs
- System `ffmpeg`
- Core Python packages:
  - `torch`
  - `transformers`
  - `qwen-vl-utils`
  - `ffmpeg-python`
  - `numpy`
  - `tqdm`
  - `accelerate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you use this benchmark or build upon this repository, please cite:

```bibtex
@article{wang2025streameqa,
  title={StreamEQA: Towards Streaming Video Understanding for Embodied Scenarios},
  author={Yifei Wang and Zhenkai Li and Tianwen Qian and Huanran Zheng and Zheng Wang and Yuqian Fu and Xiaoling Wang},
  journal={arXiv preprint arXiv:2512.04451},
  year={2025}
}
```

## Acknowledgments
 Supported by Shanghai General AI Foundation Models Program (Grant No. 2025SHZDZX025G16)

## Contact

For questions or issues, please open an issue in this repository.
