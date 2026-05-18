import random

from tqdm import tqdm

from streameqa.data import save_json
from streameqa.prompts import format_prompt, normalize_choice
from streameqa.utils.video import prepare_online_clip


def _model_name(model):
    name = getattr(model, "name", None)
    if callable(name):
        return name()
    if name:
        return name
    return model.__class__.__name__


def _model_run(model, clip, prompt):
    if hasattr(model, "run"):
        return model.run(clip, prompt)
    return model.Run(clip, prompt)


class InferOnlineBench:
    def __init__(
        self,
        video_root,
        clip_cache=None,
        context_time=64,
        width=360,
        height=420,
        skip_existing=True,
    ):
        self.video_root = video_root
        self.clip_cache = clip_cache
        self.context_time = context_time
        self.width = width
        self.height = height
        self.skip_existing = skip_existing

    def _select_items(self, data):
        items = list(data)
        random.shuffle(items)
        return items

    def eval(self, data, model, output_path):
        model_name = _model_name(model)
        items = self._select_items(data)
        processed = 0
        skipped = 0
        for item in tqdm(items, desc=f"Evaluating {model_name}"):
            if self.skip_existing and item.get(model_name):
                skipped += 1
                continue
            print("qustion id:", item.get("id"))
            clip = prepare_online_clip(
                item,
                self.context_time,
                self.video_root,
                self.clip_cache,
                width=self.width,
                height=self.height,
            )
            if not clip:
                skipped += 1
                continue
            try:
                prompt, options = format_prompt(item)
            except ValueError:
                skipped += 1
                continue
            print(f"{model_name} analyzing")
            print(f"input: {prompt}")
            response = _model_run(model, clip, prompt)
            choice = normalize_choice(response, options)
            item[model_name] = choice or response
            if choice:
                print("===========>The best option is: " + choice)
            else:
                print("===========>Model response: " + str(response))
            processed += 1
            save_json(data, output_path)
        return {"selected": len(items), "processed": processed, "skipped": skipped}
