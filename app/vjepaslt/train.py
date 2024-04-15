import definitions
import multiprocessing
import os

import numpy as np
import torch

from src.utils.logging import get_logger
from src.utils.distributed import init_distributed

from app.vjepaslt.utils import init_video_model

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

log_timings = True
log_freq = 10
checkpoint_freq = 1

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)


def main(args, resume_preempt=False):
    # TODO: Basically read the config file.

    # -- META
    cfg_meta = args.get("meta")
    seed = cfg_meta.get("seed", _GLOBAL_SEED)

    # -- MODEL
    cfg_model = args.get("model")
    model_name = cfg_model.get("model_name")
    pred_depth = cfg_model.get("pred_depth")
    pred_embed_dim = cfg_model.get("pred_embed_dim")
    uniform_power = cfg_model.get("uniform_power", True)
    use_mask_tokens = cfg_model.get("use_mask_tokens", True)
    zero_init_mask_tokens = cfg_model.get("zero_init_mask_tokens", True)

    # -- DATA
    cfg_data = args.get("data")
    dataset_type = cfg_data.get("dataset_type", "videodataset")
    mask_type = cfg_data.get("mask_type", "multiblock3d")
    dataset_paths = cfg_data.get("datasets", list())
    datasets_weights = cfg_data.get("datasets_weights", None)
    if datasets_weights:
        assert len(datasets_weights) == len(
            dataset_paths), "Must have a weighting factor for every dataset"
    batch_size = cfg_data.get("batch_size")
    num_clips = cfg_data.get("num_clips")
    num_frames = cfg_data.get("num_frames")
    tubelet_size = cfg_data.get("tubelet_size")
    sampling_rate = cfg_data.get("sampling_rate")
    duration = cfg_data.get("clip_duration", None)
    crop_size = cfg_data.get("crop_size", 224)
    patch_size = cfg_data.get("patch_size")
    pin_mem = cfg_data.get("pin_mem", False)
    num_workers = cfg_data.get("num_workers", 1)
    filter_short_videos = cfg_data.get("filter_short_videos", False)
    decode_one_clip = cfg_data.get("decode_one_clip", True)
    log_resource_util_data = cfg_data.get("log_resource_utilization", False)

    # -- LOGGING
    cfg_logging = args.get("logging")
    log_folder = cfg_logging.get("folder")
    if not cfg_logging.get("use_relative_folder", False):
        log_folder = os.path.join(
            definitions.ROOT_DIR, log_folder)
    tag = cfg_logging.get("write_tag")

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        multiprocessing.set_start_method("spawn")
    except:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    encoder, predictor = init_video_model(
        device=device,
    )
