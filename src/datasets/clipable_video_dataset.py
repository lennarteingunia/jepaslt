from abc import ABC, abstractmethod
import importlib
from typing import Optional, Type

import lightning
import pandas as pd
import torch
import torch.utils

class ClipableVideoDataset(ABC, torch.utils.data.Dataset):

    def __init__(
        self,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None
    ) -> None:
        torch.utils.data.Dataset.__init__(self)
        ABC.__init__(self)
        self.meta = self.init_meta()
        self.meta = self.filter_meta(self.meta, min_len, max_len)

    @abstractmethod
    def init_meta(self) -> pd.DataFrame:
        raise NotImplementedError()

    def filter_meta(self, meta: pd.DataFrame, min_len: int, max_len: int):
        if min_len:
            meta = meta[meta.num_frames >= min_len]
        if max_len:
            meta = meta[meta.num_frames <= max_len]
        return meta


class ClipWrapper(torch.utils.data.Dataset):

    def __init__(
        self,
        num_clips: int,
        frames_per_clip: int,
        frame_skip: int,
        dataset,
        dataset_cfg: dict,
    ) -> None:
        super(ClipWrapper, self).__init__()
        self.frame_skip = frame_skip
        self.num_clips = num_clips
        self.clip_length = frames_per_clip + frames_per_clip * frame_skip - frame_skip
        min_video_len = self.clip_length + num_clips - 1
        dataset_cfg['min_len'] = min_video_len
        dataset_module, _, dataset_name = dataset.rpartition('.')
        dataset_module = importlib.import_module(dataset_module)
        dataset_type = getattr(dataset_module, dataset_name)
        self.dataset = dataset_type(**dataset_cfg)

    def __len__(self):
        return len(self.dataset) * self.num_clips

    def __getitem__(self, idx: int) -> dict:
        # Load the full length video
        sample = self.dataset[idx]
        return clipify(sample, self.clip_length, self.num_clips, self.frame_skip)


def clipify(x: torch.Tensor, clip_length: int, num_clips: int, frame_skip: int) -> torch.Tensor:
    # What is the latest position a clip can start in a given video?
    max_clip_start_idx = len(x) - clip_length

    # Randomize a set of starting indices, a total number of num_clips
    start_idxs = torch.randperm(max_clip_start_idx + 1)[:num_clips]

    # Calculate those clips given a starting index a clip length and a number of frames to be skipped
    clips = [x[start_idx:start_idx + clip_length:frame_skip + 1]
             for start_idx in start_idxs]

    # Simply stack!
    return torch.stack(clips)
