import os

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from streameqa.models.base import ModelAdapter


class Qwen3VL(ModelAdapter):
    def __init__(
        self,
        model_path=None,
        torch_dtype="auto",
        device_map="auto",
        max_pixels=360 * 420,
        fps=1.0,
        max_new_tokens=16,
    ):
        self.model_path = model_path or os.environ.get("STREAMEQA_QWEN3VL_PATH")
        if not self.model_path:
            raise ValueError("Qwen3VL requires --model_path or STREAMEQA_QWEN3VL_PATH")
        self.max_pixels = max_pixels
        self.fps = fps
        self.max_new_tokens = max_new_tokens
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    @property
    def name(self):
        return "Qwen3VL"

    def run(self, video_file, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_file}",
                        "max_pixels": self.max_pixels,
                        "fps": self.fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

