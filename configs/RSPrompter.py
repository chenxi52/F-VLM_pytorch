from detectron2.model_zoo import get_config

default_config = get_config('configs/RSPrompter_anchor.yaml')
model = default_config['model']
model.fpn.in_channels.update = [1280] * 32