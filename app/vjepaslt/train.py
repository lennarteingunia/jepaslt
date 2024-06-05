import os
import warnings

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    warnings.warn(f"Could not set $CUDA_VISIBLE_DEVICES to $SLURM_LOCALID. This could mean many things, most likely you are not training on a slurm cluster though.")

import torch
import multiprocessing

from typing import Any, Dict
from app.vjepaslt.utils import init_encoder, init_predictor

from src.utils.distributed import init_distributed


def main(args: Dict[str, Any], resume_preempt: bool = False) -> None:

    # -- PRETRAINING ARGUMENTS
    args_pretrain = args.get('pretrain')

    args_pretrain_enc = args_pretrain.get('encoder')
    checkpoint_key = args_pretrain_enc.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain_enc.get('model_name', None)
    patch_size = args_pretrain_enc.get('patch_size', None)
    pretrain_folder = args_pretrain_enc.get('folder', None)
    enc_ckpt_fname = args_pretrain_enc.get('checkpoint', None)
    enc_use_sdpa = args_pretrain_enc.get('use_sdpa', True)
    use_SiLU = args_pretrain_enc.get('use_silu', False)
    tight_SiLU = args_pretrain_enc.get('tight_silu', True)
    uniform_power = args_pretrain_enc.get('uniform_power', False)
    enc_ckpt_path = os.path.join(pretrain_folder, enc_ckpt_fname)
    tubelet_size = args_pretrain_enc.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain_enc.get('frames_per_clip', 16)

    args_pretrain_pred = args_pretrain.get('predictor')
    pred_ckpt_fname = args_pretrain_pred.get('checkpoint', None)
    pred_depth = args_pretrain_pred.get('depth', 6)
    pred_embed_dim = args_pretrain_pred.get('pred_embed_dim')
    pred_use_sdpa = args_pretrain_pred.get('use_sdpa', True)
    pred_ckpt_path = os.path.join(pretrain_folder, pred_ckpt_fname)

        # device=device,
        # pretrained_checkpoint_path=pretrained_checkpoint_path,
        # patch_size=patch_size,
        # frames_per_clip=pretrain_frames_per_clip,
        # tubelet_size=tubelet_size,
        # crop_size=resolution,
        # depth=predictor_depth,
        # num_heads=encoder.backbone.num_heads,
        # encoder_embed_dim=encoder.backbone.embed_dim,
        # embed_dim=predictor_embed_dim,
        # uniform_power=uniform_power,
        # use_mask_tokens=use_mask_tokens,
        # num_mask_tokens=num_mask_tokens, # TODO: len(cfgs_mask)
        # zero_init_mask_tokens=zero_init_mask_tokens,
        # use_sdpa=use_sdpa

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
        pretrained_checkpoint_path=enc_ckpt_path,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=resolution,
        frames_per_clip=pretrain_frames_per_clip,
        tubelet_size=tubelet_size,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=enc_use_sdpa,
        checkpoint_key=checkpoint_key
    )

    predictor = init_predictor(
        device=device,
        pretrained_checkpoint_path=pred_ckpt_path,
        patch_size=patch_size,
        frames_per_clip=pretrain_frames_per_clip,
        tubelet_size=tubelet_size,
        crop_size=resolution,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        encoder_embed_dim=encoder.backbone.embed_dim,
        embed_dim=pred_embed_dim,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=num_mask_tokens, # TODO: len(cfgs_mask)
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=enc_use_sdpa
    )
