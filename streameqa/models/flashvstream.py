import requests
from decord import VideoReader, cpu
import torch
import sys
import os

"""
Support two Flash-VStream variants via environment variables:
- FLASH_VSTREAM_VARIANT: 'llava' or 'qwen' (default: 'llava')
- FLASH_VSTREAM_MODEL_PATH: override model weights path
- FLASH_VSTREAM_QWEN_BASE_PATH: optional base path for Qwen processor

We inject appropriate source roots into sys.path to avoid pip install.
"""
VARIANT = os.environ.get("FLASH_VSTREAM_VARIANT", "llava").lower()

if VARIANT == "llava":
    _flash_vstream_parent = os.environ.get("FLASH_VSTREAM_LLAVA_ROOT", "/path/to/Flash-VStream-LLaVA")
    if _flash_vstream_parent not in sys.path:
        sys.path.append(_flash_vstream_parent)

    from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from flash_vstream.conversation import conv_templates, SeparatorStyle
    from flash_vstream.model.builder import load_pretrained_model
    from flash_vstream.utils import disable_torch_init
    from flash_vstream.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
else:
    _flash_vstream_qwen_root = os.environ.get("FLASH_VSTREAM_QWEN_ROOT", "/path/to/Flash-VStream-Qwen")
    if _flash_vstream_qwen_root not in sys.path:
        sys.path.append(_flash_vstream_qwen_root)
    from models.vstream_qwen2vl_model import (
        FlashVStreamQwen2VLModel,
        FlashVStreamQwen2VLConfig,
        FlashVStreamQwen2VLProcessor,
        DEFAULT_FLASH_MEMORY_CONFIG,
    )
    from qwen_vl_utils import process_vision_info

from torch.multiprocessing import Process, Queue, Manager
from transformers import TextStreamer

from streameqa.models.base import ModelAdapter as Model
class FlashVstream(Model):
    def __init__(self):
        FlashVstream_Init()

    def Run(self, file, inp):
        return FlashVstream_Run(file, inp)
    
    def name(self):
        return "Flash-VStream"

def FlashVstream_Init():
    global tokenizer, model, image_processor, context_len, processor
    model_path = os.environ.get("FLASH_VSTREAM_MODEL_PATH", "/path/to/FlashVstream")

    if VARIANT == "llava":
        model_name = get_model_name_from_path(model_path)
        model_base = None
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, device="cuda", device_map="auto"
        )
        processor = None
        print(f"[Flash-VStream LLaVA] Model initialized from {model_path}.")
    else:
        # Qwen variant
        use_flash_attn = True
        model_config = FlashVStreamQwen2VLConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if getattr(model_config.vision_config, 'flash_memory_config', None) is None:
            model_config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG
        model = FlashVStreamQwen2VLModel.from_pretrained(
            model_path,
            config=model_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        ).eval()
        # For Qwen, processor may load from Qwen base if provided
        qwen_base_path = os.environ.get("FLASH_VSTREAM_QWEN_BASE_PATH", model_path)
        processor = FlashVStreamQwen2VLProcessor.from_pretrained(qwen_base_path)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else None
        image_processor = None
        context_len = None
        print(f"[Flash-VStream Qwen] Model initialized from {model_path}, processor from {qwen_base_path}.")

def load_video(video_path):
    vr = VideoReader(video_path)
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def FlashVstream_Run(file, inp):
    if VARIANT == "llava":
        video = load_video(file)
        video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
        video = [video]

        qs = inp
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                do_sample=True,
                temperature=0.002,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(outputs)
        return outputs
    else:
        # Qwen variant: use processor and message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{file}"},
                    {"type": "text", "text": inp},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            flash_memory_config=getattr(model.config.vision_config, 'flash_memory_config', None),
        )
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()
        pixel_values_videos = inputs.pixel_values_videos.cuda()
        video_grid_thw = inputs.video_grid_thw.cuda()
        visual_position_ids = inputs.visual_position_ids.cuda()

        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                max_new_tokens=128,
                top_k=1,
                do_sample=False,
                visual_position_ids=visual_position_ids,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output = outputs[0].strip()
        print(output)
        return output
