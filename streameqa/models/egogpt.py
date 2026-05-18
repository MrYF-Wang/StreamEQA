###  -----------------  ###
# Standard library imports
import copy
import os
import re
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import shutil
import threading
from datetime import datetime

import gradio as gr
import librosa

# Third-party imports
import numpy as np
import requests
# import spaces
import torch
import torch.distributed as dist
import uvicorn
import whisper
from decord import VideoReader, cpu
from egogpt.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SPEECH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    SPEECH_TOKEN_INDEX,
)
from egogpt.conversation import SeparatorStyle, conv_templates
from egogpt.mm_utils import get_model_name_from_path, process_images

# Local imports
from egogpt.model.builder import load_pretrained_model
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from streameqa.models.base import ModelAdapter as Model
import pdb
import socket

def find_free_port():
    """Find a free port dynamically"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup(rank, world_size):
    """Initialize distributed process group with dynamic port"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

class EgoGPT(Model):
    def __init__(self):
        setup(0, 1)
        # 加载模型和 tokenizer
        # 注意: device_map="auto" 对EgoGPT多模态架构兼容性差，使用单卡模式更稳定
        # 如需多卡加速，请使用多实例并行 (不同GPU运行不同CUDA_VISIBLE_DEVICES)
        model_path = os.environ.get("EGOGPT_MODEL_PATH", "/path/to/EgoGPT-7b-EgoIT")
        self.tokenizer, self.model, self.max_length = load_pretrained_model(model_path, device_map="cuda")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()

    def Run(self, file, inp):
        # 执行推理
        result_dict = generate_text(self.model, self.tokenizer, file, inp, device="cuda")

        # response = parse_model_response(result_dict["response"], ground_truth)
        response = result_dict[0]
        # pdb.set_trace()

        return response
    
    def name(self):
        return "EgoGPT"

# ================== 响应解析函数 ==================
def parse_model_response(response: str, ground_truth: str):
    prediction = ""
    reason = ""
    success = False
    parsed_status = "unknown"
    raw_response = response  # 默认保留原始响应

    try:
        response_json = json.loads(response)
        prediction = response_json.get("prediction", "")
        reason = response_json.get("reason", "")
        success = (prediction.lower() == ground_truth.lower())
        parsed_status = "primary_success"

    except Exception as pe:
        parsed_status = f"primary_error: {str(pe)}"

        try:
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
                prediction = response_json.get("prediction", "")
                reason = response_json.get("reason", "")
                success = (prediction.lower() == ground_truth.lower())
                parsed_status = "secondary_success"

        except json.JSONDecodeError as je:
            parsed_status = f"{parsed_status}, secondary_error: {str(je)}"
        except Exception as e:
            parsed_status = f"{parsed_status}, secondary_error: {str(e)}"

    result = {
        "prediction": prediction,
        "reason": reason,
        "success": success,
        "parsed_status": parsed_status
    }

    if not success or "error" in parsed_status:
        result["raw_response"] = raw_response

    return result

# # 视频加载函数
def load_video(
    video_path: Optional[str] = None,
    max_frames_num: int = 16,
    fps: int = 1,
    video_start_time: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    time_based_processing: bool = False,
) -> tuple:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    target_sr = 16000

    # Process video frames first
    if time_based_processing:
        # Initialize video reader
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()

        # Convert time to frame index based on the actual video FPS
        video_start_frame = int(time_to_frame_idx(video_start_time, video_fps))
        start_frame = int(time_to_frame_idx(start_time, video_fps))
        end_frame = int(time_to_frame_idx(end_time, video_fps))

        print("start frame", start_frame)
        print("end frame", end_frame)

        # Ensure the end time does not exceed the total frame number
        if end_frame - start_frame > total_frame_num:
            end_frame = total_frame_num + start_frame

        # Adjust start_frame and end_frame based on video start time
        start_frame -= video_start_frame
        end_frame -= video_start_frame
        start_frame = max(0, int(round(start_frame)))  # 确保不会小于0
        end_frame = min(total_frame_num, int(round(end_frame)))  # 确保不会超过总帧数
        start_frame = int(round(start_frame))
        end_frame = int(round(end_frame))

        # Sample frames based on the provided fps (e.g., 1 frame per second)
        frame_idx = [
            i
            for i in range(start_frame, end_frame)
            if (i - start_frame) % int(video_fps / fps) == 0
        ]

        # Get the video frames for the sampled indices
        video = vr.get_batch(frame_idx).asnumpy()
    else:
        # Original video processing logic
        total_frame_num = len(vr)
        # avg_fps = round(vr.get_avg_fps() / fps)
        avg_fps = 1
        try:
            frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
        except:
            pdb.set_trace()

        if max_frames_num > 0:
            if len(frame_idx) > max_frames_num:
                uniform_sampled_frames = np.linspace(
                    0, total_frame_num - 1, max_frames_num, dtype=int
                )
                frame_idx = uniform_sampled_frames.tolist()

        video = vr.get_batch(frame_idx).asnumpy()

    # Try to load audio, return None for speech if failed
    try:
        if time_based_processing:
            y, _ = librosa.load(video_path, sr=target_sr)
            start_sample = int(start_time * target_sr)
            end_sample = int(end_time * target_sr)
            speech = y[start_sample:end_sample]
        else:
            speech, _ = librosa.load(video_path, sr=target_sr)

        # Process audio if it exists
        speech = whisper.pad_or_trim(speech.astype(np.float32))
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])

        return video, speech, speech_lengths, True  # True indicates real audio

    except Exception as e:
        print(f"Warning: Could not load audio from video: {e}")
        # Create dummy silent audio
        duration = 10  # 10 seconds
        speech = np.zeros(duration * target_sr, dtype=np.float32)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])
        return video, speech, speech_lengths, False  # False indicates no real audio


# 图像加载函数
def load_images(image_paths):
    speech = torch.zeros(3000, 128)
    speech_lengths = torch.LongTensor([3000])
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        image = Image.open(path).convert("RGB")
        images.append(np.array(image))
    return np.array(images), speech, speech_lengths


# 文本分割函数
def split_text(text, keywords):
    pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
    parts = re.split(pattern, text)
    parts = [part for part in parts if part]
    return parts


# ================== 模型推理 ==================
def generate_text(model, tokenizer, video_path, prompt, device):
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    max_frames_num = 64
    fps = 1
    conv_template = "qwen_1_5"
    
    video, speech, speech_lengths, has_real_audio = load_video(
        video_path=video_path,
        max_frames_num=max_frames_num,
        fps=fps,
    )

    # Prepare the prompt based on whether we have real audio
    if not has_real_audio:
        question = f"<image>\n{prompt}"  # Video-only prompt
    else:
        question = f"<speech>\n<image>\n{prompt}"  # Video + speech prompt

    speech = torch.stack([speech]).to("cuda").half()
    processor = model.get_vision_tower().image_processor
    processed_video = processor.preprocess(video, return_tensors="pt")[
        "pixel_values"
    ]
    image = [(processed_video, video[0].size, "video")]
    image_tensor = [image[0][0].half()]
    image_sizes = [image[0][1]]
    modalities = ["video"]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    parts = split_text(prompt_question, ["<image>", "<speech>"])
    input_ids = []
    for part in parts:
        if "<image>" == part:
            input_ids += [IMAGE_TOKEN_INDEX]
        elif (
            "<speech>" == part and speech is not None
        ):  # Only add speech token if we have audio
            input_ids += [SPEECH_TOKEN_INDEX]
        else:
            input_ids += tokenizer(part).input_ids

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    generate_kwargs = {"eos_token_id": tokenizer.eos_token_id}

    result = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        speech=speech,
        speech_lengths=speech_lengths,
        do_sample=False,
        temperature=0.7,
        max_new_tokens=512,
        repetition_penalty=1.2,
        modalities=modalities,
        streamer=streamer,
        **generate_kwargs,
    )
    text_outputs = tokenizer.batch_decode(result, skip_special_tokens=True)
    # pdb.set_trace()
    return text_outputs
