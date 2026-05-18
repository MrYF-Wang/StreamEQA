import os
import subprocess
from pathlib import Path

import ffmpeg

MAX_FRAMES = 64


def build_video_path(video_id, video_root):
    if not video_id:
        return None
    return Path(video_root) / video_id.split("-")[0] / f"{video_id}.mp4"


def parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if ":" in value:
            total = 0.0
            for part in value.split(":"):
                total = total * 60.0 + float(part)
            return total
        return float(value)
    return None


def compute_online_window(timestamp, context_time, min_window_seconds=1):
    ts = parse_timestamp(timestamp)
    if ts is None:
        return None
    context = float(context_time)
    start = max(0.0, ts - context) if context > 0 else 0.0
    end = ts
    if int(ts) - int(start) <= min_window_seconds:
        return None
    if end <= start:
        return None
    return start, end


_ENCODER_CACHE = None


def _has_encoder(name):
    global _ENCODER_CACHE
    if _ENCODER_CACHE is None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
            _ENCODER_CACHE = (result.stdout or "") + (result.stderr or "")
        except FileNotFoundError:
            _ENCODER_CACHE = ""
    return name in _ENCODER_CACHE


def split_video(
    video_file,
    start_time,
    end_time,
    clip_cache=None,
    width=360,
    height=420,
):
    video_file = Path(video_file)
    if clip_cache:
        output_dir = Path(clip_cache)
    else:
        output_dir = video_file.parent / "tmp_64"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{video_file.stem}_{int(start_time)}_{end_time}.mp4"
    if output_file.exists():
        print(f"Video file {output_file} already exists.")
        return str(output_file.resolve())

    duration = max(0, int(end_time) - int(start_time))
    fps = 1.0 if duration <= MAX_FRAMES else MAX_FRAMES / duration
    print(f"Video: {output_file} ({fps} fps) is being processed.")
    output_kwargs = {
        "t": duration,
        "vf": f"fps={fps},scale={width}:{height}",
        "an": None,
        "movflags": "+faststart",
    }
    if _has_encoder("libx264"):
        output_kwargs.update({"vcodec": "libx264", "preset": "ultrafast", "crf": 28})
    elif _has_encoder("libopenh264"):
        output_kwargs.update({"vcodec": "libopenh264"})

    try:
        (
            ffmpeg.input(str(video_file), ss=int(start_time))
            .output(str(output_file), **output_kwargs)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg failed for {video_file}: {stderr}") from exc
    print(f"Video: {output_file} splitting completed.")
    return str(output_file.resolve())


def prepare_online_clip(item, context_time, video_root, clip_cache=None, width=360, height=420):
    video_path = build_video_path(item.get("video"), video_root)
    if not video_path or not os.path.isfile(video_path):
        return None
    window = compute_online_window(item.get("realtime"), context_time)
    if not window:
        return None
    start, end = window
    return split_video(video_path, start, end, clip_cache, width=width, height=height)
