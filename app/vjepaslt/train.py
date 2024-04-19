import time
import fasttext
import fasttext.util

import timm.utils
from app.vjepa.transforms import make_transforms
from app.vjepa.utils import load_checkpoint
from src.datasets.data_manager import init_data
import definitions
import multiprocessing
import os

import numpy as np
import torch
import timm

from src.utils.logging import CSVLogger, get_logger
from src.utils.distributed import init_distributed
from src.masks.multiblock3d import MaskCollator
from src.utils.tensors import repeat_interleave_batch

from app.vjepaslt.utils import init_opt, init_video_model

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
    # -- READ IN CONFIG FILE

    # -- META
    cfg_meta = args.get("meta")
    seed = cfg_meta.get("seed", _GLOBAL_SEED)
    read_checkpoint = cfg_meta.get("read_checkpoint", None)
    load_model = cfg_meta.get("load_checkpoint") or resume_preempt
    skip_batches = cfg_meta.get("skip_batches", -1)

    # Determine wether or not to use mixed precision training.
    which_dtype = cfg_meta.get("dtype")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

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

    # -- DATA AUGMENTATION
    cfgs_data_augmentation = args.get("data_aug")
    aspect_ratio_range = cfgs_data_augmentation.get(
        "random_resize_aspect_ratio", [3/4, 4/3])
    random_resize_scale = cfgs_data_augmentation.get(
        "random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_augmentation.get("motion_shift", False)
    reprob = cfgs_data_augmentation.get("reprob", 0.)
    use_auto_augmentation = cfgs_data_augmentation.get("auto_augment", False)

    # -- MASKS
    cfgs_mask = args.get("mask")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    iterations_per_epoch = cfgs_opt.get("ipe", None)
    iterations_per_epoch_scale = cfgs_opt.get("ipe_scale", 1.0)
    clip_grad = cfgs_opt.get("clip_grad", None)
    weight_decay = float(cfgs_opt.get("weight_decay"))
    final_weight_decay = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1e-8)

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

    # -- log/checkpointing paths
    log_file = os.path.join(log_folder, f"{tag}_r{rank}.csv")
    latest_file = f"{tag}-latest.pth.tar"
    latest_path = os.path.join(log_folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(
            log_folder, read_checkpoint) if read_checkpoint else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'loss-jepa'),
        ('%.5f', 'reg-loss'),
        ('%.5f', 'enc-grad-norm'),
        ('%.5f', 'pred-grad-norm'),
        ('%d', 'gpu-time(ms)'),
        ('%d', 'wall-time(ms)'),
    )

    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        # TODO: What is the number of mask tokens?
        zero_init_mask_tokens=zero_init_mask_tokens,
        # TODO: What is sdpa and why should I be using it?
    )
    target_encoder = timm.utils.ModelEmaV3(model=encoder)

    # logger.info("Downloading and initializing fasttext models.")
    # fasttext.util.download_model("en", if_exists="ignore")
    # embedder = fasttext.load_model("cc.en.300.bin")

    if mask_type == "multiblock3d":
        logger.info("Initializing basic multi-block masking.")
        mask_collator = MaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask,
        )
    else:
        logger.info("Initializing random tube masking.")
        mask_collator = MaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask,
        )

    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=aspect_ratio_range,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=use_auto_augmentation,
        motion_shift=motion_shift,
        crop_size=crop_size
    )

    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        filter_short_videos=filter_short_videos,
        decode_one_clip=decode_one_clip,
        duration=duration,
        num_clips=num_clips,
        transform=transform,
        datasets_weights=datasets_weights,
        collator=mask_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        rank=rank,
        log_dir=log_folder if log_resource_util_data else None
    )

    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        _dlen = unsupervised_loader.num_batches
    if not iterations_per_epoch:
        iterations_per_epoch = _dlen
    logger.info(
        f"Iterations per epoch/dataset lenght: {iterations_per_epoch}/{_dlen}")

    optimizer, scaler, scheduler, weight_decay_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        iterations_per_epoch=iterations_per_epoch,
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        final_weight_decay=final_weight_decay,
        final_lr=final_lr,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps
    )
    encoder = torch.nn.parallel.DistributedDataParallel(
        encoder, static_graph=True)
    predictor = torch.nn.parallel.DistributedDataParallel(
        predictor, static_graph=True)
    target_encoder = torch.nn.parallel.DistributedDataParallel(
        target_encoder, static_graph=True)

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i * (ema[1] - ema[0]) / (iterations_per_epoch * num_epochs * iterations_per_epoch_scale)
                          for i in range(int(iterations_per_epoch * num_epochs * iterations_per_epoch_scale) + 1))

    start_epoch = 0

    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=load_path, encoder=encoder, predictor=predictor, target_encoder=target_encoder, opt=optimizer, scaler=scaler)

        # -- "Loading" all schedulers by simply stepping through them until the desired point
        for _ in range(start_epoch * iterations_per_epoch):
            scheduler.step()
            weight_decay_scheduler.stepp()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": scaler if not scaler else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            # TODO: Currently missing loss average meter
            "batch-_size": batch_size,
            "world_size": world_size,
            "lr": lr
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing data loader.")
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skipping {skip_batches} batches.")
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skipped {itr}/{skip_batches} batches.")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        unsupervised_sampler.set_epoch(epoch)

        for itr in range(iterations_per_epoch):
            itr_start_time = time.time()

            try:
                udata, masks_enc, masks_pred = next(loader)
            except Exception:
                logger.info("Exhaused data loaders. Refreshing...")
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred = next(loader)
            assert len(masks_enc) == len(
                masks_pred), "Currently require the number of encoder masks to be the number of predictor masks."

            def load_clips_and_words():
                # Put each clip on the GPU and concat along batch dimension
                clips = torch.cat([u.to(device, non_blocking=True)
                                  for u in udata[0]], dim=0)

                
                # Put each mask pair on the GPU and reuse the same mask pair for each clip
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(
                        _me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(
                        _mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return clips, _masks_enc, _masks_pred

            clips, masks_enc, masks_pred = load_clips_and_words()

            clips = encoder(clips, masks_enc)
