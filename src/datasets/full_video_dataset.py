from __future__ import annotations

import logging
import os
from typing import Any, List
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
        logger=logger
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
        transform,
        logger: logging.Logger = logging.getLogger(__name__)
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

        for video_path, label in tqdm.tqdm(zip(video_paths, labels), total=len(video_paths), desc='Indexing clips...'):

            video_reader = decord.VideoReader(
                video_path, num_threads=-1, ctx=decord.cpu(0))
            num_clips = len(video_reader) - self.clip_len

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

        return (
            [self.transform(clip)],
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

    dataset = VideoDataset(
        ['/mnt/datasets/CREMA-D/additional/splits/split_0.csv']
    )
    for buffer, label, clip_indices in dataset:
        for clip in buffer:
            print(clip.shape)
        print(label)
        print(clip_indices)
        break

    dataset = FullVideoDataset(
        ['/mnt/datasets/CREMA-D/additional/splits/split_0.csv']
    )
    print(f'Dataset Length is: {len(dataset)}')
    for clip, label, meta in tqdm.tqdm(dataset):
        print(clip.shape)
        print(label)
        print(meta['indices'])
        break
