import argparse
import logging
import os
from typing import Any, Dict
import torch
import torch.amp
import torch.multiprocessing.spawn
import torch.multiprocessing.spawn
import torch.nn.functional as F

from evals.video_classification_frozen.eval import make_dataloader
from evals.video_classification_frozen.utils import ClipAggregation
from experiments import build_attentive_classifier, build_encoder, load_config
from utils.distributed import AllReduce


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
):
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__ + str(rank))

    logger.info('Setting up distributed data paralell execution.')
    ddp_setup(rank, world_size)

    # logger.info(f'Building the encoder model')
    # encoder = build_encoder(config=config).to(rank)
    # encoder.eval()
    # for p in encoder.parameters():
    #     p.requires_grad = False
    # encoder = ClipAggregation(encoder, tubelet_size=2, attend_across_segments=config.get(
    #     'optimization').get('attend_across_segments', False))

    # logger.info(f'Building the classifier model')
    # classifier = build_attentive_classifier(
    #     config, checkpoint_path=classifier_checkpoint_path).to(rank)
    # classifier.eval()
    # for p in classifier.parameters():
    #     p.requires_grad = False

    logger.info(f'Building the data loader.')
    val_dataloader = make_dataloader(
        dataset_type=config.get('data').get('dataset_type', 'VideoDataset'),
        root_path=config.get('data').get('dataset_val'),
        resolution=config.get('optimization').get('resolution', 224),
        frames_per_clip=config.get('data').get('frames_per_clip', 16),
        frame_step=config.get('pretrain').get('frame_step', 4),
        num_segments=config.get('data').get('num_segments', 1),
        eval_duration=config.get('pretrain').get('clip_duration', None),
        num_views_per_segment=config.get(
            'data').get('num_views_per_segment', 1),
        allow_segment_overlap=True,
        batch_size=config.get('optimization').get('batch_size'),
        world_size=world_size,
        rank=rank,
        training=False
    )

    logger.info(f'{len(val_dataloader)=}')
    # for itr, data in enumerate(val_dataloader):

    #     with torch.cuda.amp.autocast(dtype=torch.float16, enabled=config.get('optimization').get('use_bfloat16')):

    #         # Load data and put on GPU
    #         clips = [
    #             [dij.to(rank, non_blocking=True)
    #              for dij in di]  # iterate over spatial views of clip
    #             for di in data[0]  # iterate over temporal index of clip
    #         ]
    #         clip_indices = [d.to(rank, non_blocking=True) for d in data[2]]
    #         labels = data[1].to(rank)
    #         batch_size = len(labels)

    #         # Forward and prediction
    #         with torch.no_grad():
    #             outputs = encoder(clips, clip_indices)
    #             if config.get(
    #                     'optimization').get('attend_across_segments', False):
    #                 outputs = [classifier(o) for o in outputs]
    #             else:
    #                 outputs = [[classifier(ost) for ost in os]
    #                            for os in outputs]

    #     with torch.no_grad():
    #         if config.get(
    #                 'optimization').get('attend_across_segments', False):
    #             outputs = sum([F.softmax(o, dim=1)
    #                           for o in outputs]) / len(outputs)
    #         else:
    #             outputs = sum([sum([F.softmax(ost, dim=1) for ost in os])
    #                           for os in outputs]) / len(outputs) / len(outputs[0])
    #         top1_acc = 100. * \
    #             outputs.max(dim=1).indices.eq(labels).sum() / batch_size
    #         top1_acc = float(AllReduce.apply(top1_acc))

    #     logger.info(top1_acc)

    logger.info(f'Distributed data paralell teardown.')
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run evaluation using V-JEPA.')
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--classifier-checkpoint-path',
                        type=str, required=True)
    parser.add_argument('--single-device', default=False, action='store_true')
    args = parser.parse_args()

    config = load_config(config_path=args.config_path)

    if not args.single_device:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    max_rank = world_size
    torch.multiprocessing.spawn(run_eval, args=(
        world_size, config, args.classifier_checkpoint_path), nprocs=max_rank)
