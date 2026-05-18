import os
import transformers
from videollmonline.data.utils import ffmpeg_once
from videollmonline.demo.inference import LiveInfer
from sentence_transformers import SentenceTransformer, util
import pdb

logger = transformers.logging.get_logger('liveinfer')

from streameqa.models.base import ModelAdapter as Model
class VideollmOnline(Model):
    def __init__(self):
        """
        Initialize the model
        """
        super().__init__()
        self.liveinfer = LiveInfer()
        self.MiniLM = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def Run(self, file, inp):
        """
        Given the video file and input prompt, run the model and return the response
        file: Video file path
        inp: Input prompt
        """
        timestamp = float(file.split('/')[-2].split('_')[-1])
        return self.videollmOnline_Run(file, inp, timestamp)

    @staticmethod
    def name():
        """
        Return the name of the model
        """
        return "VideollmOnline"

    def videollmOnline_Run(self, file, inp, timestamp):
        qus = inp.split("Question: ")[1].split("\n\nOptions:")[0]
        ops = []
        for item in inp.split("\n\nOptions:")[1].split("\n\nThe best option is:")[0].split("\n"):
            if item.strip() != "":
                ops.append(item.split(". ")[1])

        self.liveinfer.reset()
        
        name, ext = os.path.splitext(file)
        name = name.split('/')[-1]
        cache_root = os.environ.get("VIDEOLLMONLINE_CACHE", "/path/to/cache")
        ffmpeg_video_path = os.path.join(cache_root, name + f'_{self.liveinfer.frame_fps}fps_{self.liveinfer.frame_resolution}' + ext)
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(file, ffmpeg_video_path, fps=self.liveinfer.frame_fps, resolution=self.liveinfer.frame_resolution)
        logger.warning(f'{file} -> {ffmpeg_video_path}, {self.liveinfer.frame_fps} FPS, {self.liveinfer.frame_resolution} Resolution')

        self.liveinfer.load_video(ffmpeg_video_path)
        # self.liveinfer.load_video(file)
        timestamp = (self.liveinfer.num_video_frames - 4) / self.liveinfer.frame_fps
        self.liveinfer.input_query_stream(qus, video_time=timestamp)
        # pdb.set_trace()

        for i in range(self.liveinfer.num_video_frames):
            self.liveinfer.input_video_stream(i / self.liveinfer.frame_fps)
            query, response = self.liveinfer()

            if response:
                print(response)
                # pdb.set_trace()
                break
        if response is None:
            # pdb.set_trace()
            return "0"

        pred_idx = compare_ops(self.MiniLM, ops, response)
        idx_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
        response = idx_dict[pred_idx]
        # pdb.set_trace()
        print(response)
        return response

def compare_ops(model, ops, response):
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    ops_embeddings = model.encode(ops, convert_to_tensor=True)
    response_embedding = model.encode(response, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(response_embedding, ops_embeddings)[0]
    best_idx = cos_scores.argmax().item()
    # pdb.set_trace()
    # return ops[best_idx]
    return best_idx
