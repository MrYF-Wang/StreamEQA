from importlib import import_module


MODEL_REGISTRY = {
    "Qwen3VL": ("streameqa.models.qwen3vl", "Qwen3VL"),
    "qwen3vl": ("streameqa.models.qwen3vl", "Qwen3VL"),
    "InternVL3-8B": ("streameqa.models.internvl", "InternVL"),
    "InternVL": ("streameqa.models.internvl", "InternVL"),
    "LongVA": ("streameqa.models.longva", "LongVA"),
    "MiniCPM-V": ("streameqa.models.minicpmv", "MiniCPMV"),
    "MiniCPMV": ("streameqa.models.minicpmv", "MiniCPMV"),
    "VideoLLaMA3": ("streameqa.models.videollama3", "VideoLLaMA3"),
    "EgoGPT": ("streameqa.models.egogpt", "EgoGPT"),
    "EgoVLPv2": ("streameqa.models.egovlpv2", "EgoVLPv2"),
    "VideollmOnline": ("streameqa.models.videollmonline", "VideollmOnline"),
    "FlashVstream": ("streameqa.models.flashvstream", "FlashVstream"),
    "FlashVStream": ("streameqa.models.flashvstream", "FlashVstream"),
    "Dispider": ("streameqa.models.dispider", "Dispider"),
    "TimeChatOnline": ("streameqa.models.timechatonline", "TimeChatOnline"),
}


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{name}'. Supported models: {supported}")
    module_name, class_name = MODEL_REGISTRY[name]
    cls = getattr(import_module(module_name), class_name)
    if module_name == "streameqa.models.qwen3vl":
        return cls(**{k: v for k, v in kwargs.items() if v is not None})
    return cls()
