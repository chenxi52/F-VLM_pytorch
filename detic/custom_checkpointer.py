from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer

class samCheckpointer(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)

    def _load_file(self, filename):
        loaded = self._torch_load(filename)

        if "model" not in loaded:
            loaded = {"model": loaded}
        assert self._parsed_url_during_load is not None, "`_load_file` must be called inside `load`"
        loaded["matching_heuristics"] = True
       
        return loaded