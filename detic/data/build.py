from detectron2.data.build import get_detection_dataset_dicts, build_detection_test_loader, build_detection_train_loader, _train_loader_from_config
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler
import torch.utils.data as torchdata
from detectron2.config import configurable
from typing import Optional,List,Any,Callable
def test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(dataset))
        if not isinstance(dataset, torchdata.IterableDataset)
        else None,
        "batch_size": cfg.TEST.IMS_PER_BATCH,
        "persistent_workers": cfg.DATALOADER.PERSISTENT_WORKERS,
    }
    
@configurable(from_config=test_loader_from_config)
def custom_build_detection_test_loader(dataset,mapper,sampler,batch_size,num_workers,collate_fn: Optional[Callable[[List[Any]], Any]] = None):
    return build_detection_test_loader( 
        dataset=dataset,
        mapper=mapper,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn)

def train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    config  = _train_loader_from_config(cfg, mapper, dataset, sampler)
    config['persistent_workers'] = cfg.DATALOADER.PERSISTENT_WORKERS
    return config

@configurable(from_config=train_loader_from_config)
def custom_build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory=False,
    ):
    return build_detection_train_loader(
        dataset=dataset,
        mapper=mapper,
        sampler=sampler,
        total_batch_size=total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )