from typing import List
import yaml
import os

import torch
# import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.distributed.parallel_loader import MpDeviceLoader

from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: list[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore
    xmp.spawn(_eval_fn, args=(eval_cfg,), nprocs=8)


def _eval_fn(rank: int, eval_cfg: EvalConfig):
    RANK = xm.get_ordinal()
    WORLD_SIZE = xm.xrt_world_size()

    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # Dataloader
    device = xm.xla_device()
    train_loader, train_metadata = create_dataloader(config, "train", rank=RANK, world_size=WORLD_SIZE, test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", rank=RANK, world_size=WORLD_SIZE, test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size)
    eval_loader = MpDeviceLoader(eval_loader, device)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    # Try unwrap torch.compile
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location="cpu"), assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location="cpu").items()}, assign=True)
    
    train_state.model = train_state.model.to(device)
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    # Evaluate
    print ("Starting evaluation")
    
    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE, device=device)

    if metrics is not None:
        print (metrics)


if __name__ == "__main__":
    launch()
