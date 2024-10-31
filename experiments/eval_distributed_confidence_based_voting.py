import argparse
import itertools
import logging
import os
from typing import Any, Dict, Optional
import decord
import pandas as pd
import torch
import sha3
import torch.amp
import torch.multiprocessing.spawn
import torch.multiprocessing.spawn
import torch.nn.functional as F
import torch.utils.data.distributed
import torchvision
import torchvision.transforms.functional
import tqdm

from datasets.full_video_dataset import make_fullvideodata
from evals.video_classification_frozen.eval import make_dataloader
from evals.video_classification_frozen.utils import ClipAggregation
from experiments import build_attentive_classifier, build_encoder, load_config
from utils.distributed import AllReduce
from utils.logging import AverageMeter


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '37137'
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl', rank=rank, world_size=world_size)


def run_eval(
    rank: int,
    world_size: int,
    config: Dict[str, Any],
    classifier_checkpoint_path: str,
    overwrite_encoder_checkpoint: Optional[str],
):
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__ + str(rank))

    logger.info('Setting up distributed data paralell execution.')
    ddp_setup(rank, world_size)

    logger.info(f'Building the encoder model')
    encoder = build_encoder(
        config=config, overwrite_checkpoint=overwrite_encoder_checkpoint).to(rank)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = ClipAggregation(encoder, tubelet_size=2, attend_across_segments=config.get(
        'optimization').get('attend_across_segments', False))

    logger.info(f'Building the classifier model')
    classifier = build_attentive_classifier(
        config, checkpoint_path=classifier_checkpoint_path).to(rank)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    logger.info(f'Building the data loader.')

    _, dataloader, _ = make_fullvideodata(
        data_paths=config.get('data').get('dataset_val'),
        batch_size=config.get('optimization').get('batch_size'),
        world_size=world_size,
        rank=rank,
        logger=logger
    )

    prediction_cls_probabilities =  {}
    for _, (clips, labels, metas) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):

        # Data transformations for inference.

        B, T, H, W, C = clips.shape
        clips = clips.permute(0, 1, 4, 2, 3)  # B, T, C, H, W
        clips = clips.reshape(B * T, C, H, W)  # B * T, C, H, W
        clips = clips.float() / 255
        crop_size = config.get('optimization').get('resolution', 224)
        clips = torchvision.transforms.functional.resize(
            clips,
            size=(crop_size, crop_size),
            interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR
        )
        clips = torchvision.transforms.functional.normalize(
            clips,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        clips = clips.reshape(B, T, C, crop_size, crop_size)  # B, T, C, H, W
        clips = clips.permute(0, 2, 1, 3, 4)  # B, C, T, H, W

        # Ordering

        clips = clips.to(rank)
        clips = [[clips]]
        clip_indices = metas['indices']
        labels = labels.to(rank)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if config.get('optimization').get('attend_across_segments', False):
                    outputs = [classifier(o) for o in outputs]
                else:
                    outputs = [[classifier(ost) for ost in os]
                               for os in outputs]

                class_probabilities = sum([F.softmax(o, dim=1)
                                           for o in outputs]) / len(outputs)

                for path, cls_prob, label in zip(metas['path'], class_probabilities, labels):
                    if path in prediction_cls_probabilities:
                        prediction_cls_probabilities[path] = {
                            'probabilities': prediction_cls_probabilities[path]['probabilities'] + cls_prob,
                            'number_pred': prediction_cls_probabilities[path]['number_pred'] + 1,
                            'label': prediction_cls_probabilities[path]['label']
                        }
                    else:
                        prediction_cls_probabilities[path] = {
                            'probabilities': cls_prob,
                            'number_pred': 1,
                            'label': label
                        }

    num_classes = config.get('data').get('num_classes', None)
    predictions = torch.zeros()
    

    logger.info(f'Distributed data paralell teardown.')
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run evaluation using V-JEPA.')
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--classifier-checkpoint-path',
                        type=str, required=True)
    parser.add_argument('--overwrite-encoder-checkpoint',
                        required=False, default=None)
    parser.add_argument('--single-device', default=False, action='store_true')
    args = parser.parse_args()

    config = load_config(config_path=args.config_path)

    if not args.single_device:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    max_rank = world_size
    torch.multiprocessing.spawn(run_eval, args=(
        world_size, config, args.classifier_checkpoint_path, args.overwrite_encoder_checkpoint), nprocs=max_rank)
