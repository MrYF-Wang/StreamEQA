from streameqa.models.base import ModelAdapter as Model
from modelscope import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import torch
import os

class VideoLLaMA3(Model):
    def __init__(self):
        self.model, self.processor = VideoLLaMA3_Init()

    def Run(self, file, inp):
        return VideoLLaMA3_Run(self.model, self.processor, file, inp)
    
    def name(self):
        return "VideoLLaMA3"

def VideoLLaMA3_Init():
    # 1. Initialize the model.
    model_path = os.environ.get("VIDEOLLAMA3_MODEL_PATH", "/path/to/VideoLLaMA3-7B")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

# def VideoLLaMA3_Run(file, inp):
#     # Video Inference
#     # Reply:
#     # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.
#     modal = 'video'

#     # 2. Visual preprocess (load & transform image or video).
#     output = mm_infer(processor[modal](file), inp, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

#     return output[0]

def VideoLLaMA3_Run(model, processor, file, inp):
    video_path = file
    question = inp

    # Video conversation
    conversation = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                # {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 128}},
                {"type": "video", "video": {"video_path": video_path}},
                {"type": "text", "text": question},
            ]
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(response)
    return response
