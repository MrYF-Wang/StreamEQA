from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import logging
import time
from collections import defaultdict
import argparse
import ffmpeg
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from timechatonline import Qwen2_5_VLForConditionalGeneration

from streameqa.models.base import ModelAdapter as Model
import pdb


class TimeChatOnline(Model):
    def __init__(self):
        # self.LOG_PATH = "log/{run_name}_{curr_time}.log"
        # self.OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
        # self.DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"

        # # Set up logging
        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        # fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
        # fmt = logging.Formatter(fmt_str)

        # Update global variables
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_NAME = "feature_0d5"
        self.DROP_METHOD = 'feature'
        self.DROP_THRESHOLD = 0.5
        self.DROP_ABSOLUTE = True

        CKPT_PATH = os.environ.get("TIMECHATONLINE_MODEL_PATH", "/path/to/TimeChatOnline-7B")
        # self.TASK_CSV = "/home/gaohuan03/yaolinli/datasets/StreamingBench/annos/Real_Time_Visual_Understanding.csv"
        # self.VIDEO_DIR = "/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding"
        # self.LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
        # self.OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
        drop_root = os.environ.get("TIMECHATONLINE_DROP_DIR", "/path/to/timechatonline/drop")
        self.DR_SAVE_PATH = os.path.join(drop_root, f"{RUN_NAME}_{curr_time}.jsonl")
        os.makedirs(self.DR_SAVE_PATH.rsplit('/',1)[0], exist_ok=True)

        self.MIN_PIXELS = 448*448
        self.MAX_PIXELS = 448*448
        self.MIN_FRAMES = 4
        self.MAX_FRAMES = 1016

        # # Create result directory
        # os.makedirs(RESULT_DIR, exist_ok=True)
        # os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
        # os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
        # os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)
        
        # # Add file handler
        # file_handler = logging.FileHandler(LOG_PATH)
        # file_handler.setFormatter(fmt)
        # file_handler.setLevel(logging.INFO)
        # self.logger.addHandler(file_handler)

        # # Print run info
        # self.logger.info(f"Running {RUN_NAME} on StreamingBench")
        # self.logger.info(f"Drop method: {DROP_METHOD}")
        # self.logger.info(f"Drop threshold: {DROP_THRESHOLD}")
        # self.logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
        # self.logger.info(f"Checkpoint path: {CKPT_PATH}")
        # self.logger.info(f"Result dir: {RESULT_DIR}")
        # self.logger.info(f"Task csv: {TASK_CSV}")
        # self.logger.info(f"Video dir: {VIDEO_DIR}")
        # self.logger.info(f"Output jsonl: {OUTPUT_JSONL}")
        # self.logger.info(f"Drop ratio info save path: {DR_SAVE_PATH}")
        # self.logger.info(f"Min pixels: {MIN_PIXELS}")
        # self.logger.info(f"Max pixels: {MAX_PIXELS}")
        # self.logger.info(f"Max frames: {MAX_FRAMES}")
        # self.logger.info(f"Min frames: {MIN_FRAMES}")

        # Load model and processor
        torch.manual_seed(1234)
        # self.logger.info(f"Set manual seed to 1234")
        ## Use Qwen2.5-VL
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            CKPT_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            CKPT_PATH,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )
        # self.logger.info(f"Load model and processor from {CKPT_PATH}")

        # Load task info
        # self.task_df = pd.read_csv(TASK_CSV)

    def Run(self, file, inp):
        return self.inference(file, inp)
    
    def name(self):
        return "TimeChatOnline"

    def inference(self, file, inp):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": file,
                        "min_pixels": self.MIN_PIXELS,
                        "max_pixels": self.MAX_PIXELS,
                        "max_frames": self.MAX_FRAMES,
                        "min_frames": self.MIN_FRAMES,
                        "fps": 1.0,
                    },
                    {
                        "type": "text", 
                        "text": inp
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(torch.device('cuda'))
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            drop_method=self.DROP_METHOD,
            drop_threshold=self.DROP_THRESHOLD,
            drop_absolute=self.DROP_ABSOLUTE,
            dr_save_path=self.DR_SAVE_PATH,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]

        print(response)
        return response
