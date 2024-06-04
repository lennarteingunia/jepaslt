import os
import warnings

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    warnings.warn(f"Could not set $CUDA_VISIBLE_DEVICES to $SLURM_LOCALID. This could mean many things, most likely you are not training on a slurm cluster though.")

import torch
import multiprocessing

from typing import Any, Dict
from app.vjepaslt.utils import init_encoder

from src.utils.distributed import init_distributed


def main(args: Dict[str, Any], resume_preempt: bool = False) -> None:

    # -- PRETRAINING ARGUMENTS
    args_pretrain = args.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_checkpoint_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- OPTIMIZATION
    args_opt = args.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')

    try:
        multiprocessing.set_start_method("spawn")
    except Exception:
        warnings.warn(f"Could not set start method to 'spawn'.")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()

    encoder = init_encoder(
        device=device,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=resolution,
        frames_per_clip=pretrain_frames_per_clip,
        tubelet_size=tubelet_size,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa
    )
