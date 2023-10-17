from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer

class samCheckpointer(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)

    def _load_file(self, filename):
        loaded = super()._load_file(filename)
        loaded["matching_heuristics"] = True
       
        return loaded