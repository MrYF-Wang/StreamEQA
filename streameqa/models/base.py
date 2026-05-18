class ModelAdapter:
    def run(self, video_file, prompt):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

