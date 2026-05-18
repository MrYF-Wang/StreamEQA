import torch
import json
import os
import time
import argparse
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from transformers import RobertaTokenizer
from egovlpv2.video_qa_model_linear_end2end import FrozenInTime

from streameqa.models.base import ModelAdapter as Model
import pdb


class EgoVLPv2(Model):
    def __init__(self):
        # 初始化模型
        video_params = {
            "model": "SpaceTimeTransformer",
            "arch_config": "base_patch16_224",
            "num_frames": 16,
            "pretrained": True,
            "time_init": "zeros"
        }
        text_params = {
            "model": "roberta-base",
            "pretrained": True,
            "input": "text"
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FrozenInTime(
            video_params, text_params,
            projection_dim=4096, # 768,
            load_checkpoint=os.environ.get("EGOVLPV2_CHECKPOINT", "/path/to/EgoVLPv2.pth"),
            output_dim=4
        ).to(self.device)
        self.model.eval()

        # 初始化 BERT tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def Run(self, file, inp):
        frames_dir = file.split('.mp4')[0]
        os.makedirs(file.split('.mp4')[0], exist_ok=True)
        cap = cv2.VideoCapture(file)
        frame_count = 0
        frames_dirs = []
        ret, frame = cap.read()
        while ret:
            frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
            frames_dirs.append(frame_filename)
            try:
                cv2.imwrite(frame_filename, frame)
            except:
                pdb.set_trace()
                continue
            frame_count += 1
            ret, frame = cap.read()
        cap.release()
        if frame_count == 0:
            return '0'
        # pdb.set_trace()

        video_tensor = load_video_frames(frames_dirs, self.transform, 16)

        # 编码问题
        input_ids, attention_mask = tokenize_question(inp, self.tokenizer)

        # 推理
        pred_idx, probs = predict(self.model, video_tensor, input_ids, attention_mask, self.device)
        idx_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
        response = idx_dict[pred_idx]
        # pdb.set_trace()

        return response
    
    def name(self):
        return "EgoVLPv2"

def load_video_frames(frame_paths, transform, num_frames=4):
    """从路径列表加载视频帧"""
    frames = []
    frame_paths = sorted(frame_paths) 
    indices = np.linspace(0, len(frame_paths) - 1, num_frames).astype(int)
    # pdb.set_trace()
    for idx in indices:
        img = Image.open(frame_paths[idx]).convert('RGB')
        frames.append(transform(img))
    return torch.stack(frames).unsqueeze(0)  # (1, T, C, H, W)


def tokenize_question(question, tokenizer, max_length=32):
    """使用 BERT tokenizer 编码问题"""
    encoded = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']


def predict(model, video_tensor, input_ids, attention_mask, device):
    """模型推理"""
    model.eval()
    with torch.no_grad():
        data_batch = {
            "video": video_tensor.to(device),
            "text": {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device)
            }
        }
        logits = model(data_batch)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        return pred_idx, probs.cpu().numpy()
