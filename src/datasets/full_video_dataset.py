from __future__ import annotations

import logging
import os
from typing import Any, List
import warnings
import numpy as np
import pandas as pd
import torch
import decord
import torch.utils.data.distributed
import tqdm
import sha3
import yaml

from datasets.video_dataset import VideoDataset


def make_fullvideodata(
    data_paths: List[str],
    batch_size: int,
    world_size: int,
    frames_per_clip: int,
    frame_step: int,
    rank: int,
    transform,
    *,
    num_workers: int = 10,
    logger: logging.Logger = logging.getLogger(__name__)
):

    dataset = FullVideoDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        transform=transform,
        logger=logger,
        min_pad=True
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return dataset, data_loader, sampler


class FullVideoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_paths: List[str],
        frames_per_clip: int = 16,
        frame_step: int = 4,
        *,
        transform=None,
        logger: logging.Logger = logging.getLogger(__name__),
        min_pad: bool = True
    ) -> None:
        super(FullVideoDataset, self).__init__()

        video_paths, labels = [], []
        for data_path in data_paths:
            data = pd.read_csv(data_path, header=None, delimiter=' ')
            video_paths += list(data.values[:, 0])
            labels += list(data.values[:, 1])

        self.clip_len = int(frames_per_clip * frame_step)
        self.logger = logger
        self.transform = transform

        # Creating indexing file path if it exists.
        id_string = ''.join(map(str, video_paths + labels))
        k = sha3.keccak_512()
        k.update(id_string.encode('utf-8'))
        index_file_id = k.hexdigest()
        index_file_name = index_file_id + '.yaml'
        # TODO: This cannot be the final location for tmp files.
        self.index_file_path = index_file_name

        self.samples = []

        num_dropped_videos = 0
        total_num_videos = 0
        p_bar = tqdm.tqdm(zip(video_paths, labels), total=len(video_paths))
        for video_path, label in p_bar:

            video_reader = decord.VideoReader(
                video_path, num_threads=-1, ctx=decord.cpu(0))
            num_clips = len(video_reader) - self.clip_len

            if num_clips < 1:
                if not min_pad:
                    logger.info(

                        f'Dropping video at {video_path}, because it does not fit a single clip')
                    num_dropped_videos += 1
                    continue

                # If min_pad is enabled we pad the clip index with the last frame of the video.

                if min_pad:

                    indices = np.linspace(
                        0, len(video_reader), num=len(video_reader) // frame_step)
                    indices = np.concatenate((indices, np.ones(
                        frames_per_clip - len(video_reader) // frame_step) * len(video_reader),))
                    indices = np.clip(indices, 0, len(
                        video_reader) - 1).astype(np.int64)

                    self.samples.append({
                        'path': video_path,
                        'indices': indices,
                        'label': label,
                        'position': start_idx,
                        'video_len': num_clips
                    })

                    total_num_videos += 1
            else:
                total_num_videos += 1

            p_bar.set_description(
                f'Indexing videos. Number of indexed videos: {total_num_videos}; Number of dropped videos: {num_dropped_videos}.')

            for start_idx in range(num_clips):

                end_idx = start_idx + self.clip_len

                indices = np.linspace(
                    start_idx,
                    end_idx,
                    num=frames_per_clip
                )
                indices = np.clip(
                    indices,
                    start_idx,
                    end_idx - 1
                ).astype(np.int64)

                self.samples.append({
                    'path': video_path,
                    'indices': indices,
                    'label': label,
                    'position': start_idx,
                    'video_len': num_clips
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> Any:
        sample = self.samples[index]

        video_reader = decord.VideoReader(
            sample['path'], num_threads=-1, ctx=decord.cpu(0))

        clip = video_reader.get_batch(sample['indices']).asnumpy()

        if self.transform is not None:
            clip = self.transform(clip)

        return (
            [clip],
            sample['label'],
            {
                'path': sample['path'],
                'indices': [sample['indices']],
                'position': sample['position'],
                'video_len': sample['video_len']
            }
        )

    def save_index_to_tmp_file(self):
        if os.path.exists(self.index_file_path):
            self.logger.info(
                f'Not saving index file to {self.index_file_path} since it already exists.')
            return
        with open(self.index_file_path, 'w+') as f:
            yaml.dump(self.samples, f)

    def load_index_from_tmp_file(self):
        with open(self.index_file_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == '__main__':

    dataset = FullVideoDataset(
        ['/mnt/datasets/CREMA-D/additional/splits/split_0.csv'],
        min_pad=True
    )
    for clip, label, meta in tqdm.tqdm(dataset):
        pass
