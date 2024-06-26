import numpy as np
import pandas as pd
import torch
import decord


class VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        frames_per_clip: int = 16,
        frame_step: int = 4,
        num_clips: int = 1,
        shared_transforms=list(),
        clip_transforms=list(),
        label_transforms=list(),
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False
    ) -> None:
        self.root = root
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.shared_transforms = shared_transforms
        self.clip_transforms = clip_transforms
        self.label_transforms = label_transforms
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap

        # TODO: What are my actual headers going to be?
        self.data = pd.read_csv(
            self.root,
            sep=';',
        )

        # Filter videos for videos that are too short.
        self.clip_len = self.frames_per_clip + self.frames_per_clip * \
            (self.frame_step - 2) - (self.frame_step - 2)
        self.data = self.data[self.data.video_length <
                              self.clip_len]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        label = sample.label
        clips, clip_indices = self._load_video(sample.fname)
        # TODO: Actually use the transforms
        return {
            'clips': clips,
            'clip_indices': clip_indices,
            'label': label
        }

    def _load_video(self, fname: str) -> torch.Tensor:
        decord.bridge.set_bridge('torch')

        vr = decord.VideoReader(fname, num_threads=-1)

        # Go to the beginning of the video
        vr.seek(0)

        # Partition the video into equal sized segments and sample each clip from a different segment
        partition_length = len(vr) // self.num_clips
        all_indices, clip_indices = [], []
        for clip_idx in range(self.num_clips):

            if partition_length > self.clip_len:
                end_idx = self.clip_len
                if self.random_clip_sampling:
                    end_idx = np.random.randint(
                        self.clip_len, partition_length)
                start_idx = end_idx - self.clip_len
                indices = np.linspace(start_idx, end_idx,
                                      num=self.frames_per_clip)
                indices = np.clip(indices, start_idx,
                                  end_idx - 1).astype(np.int64)
                # Move indices to correct location in the video
                indices = indices + clip_idx * partition_length
            else:
                # This means that partitions of the video overlap. If this is not allowed we repeatedly append the last frame in the segment until we reach the desired clip length
                if not self.allow_clip_overlap:
                    # First create the number of indices that fit into that partition
                    indices = np.linspace(
                        0, partition_length, num=partition_length // self.frame_step)

                    # Append the indices of the last frame the remaining times
                    indices = np.concatenate((indices, np.ones(
                        self.frames_per_clip - partition_length // self.frame_step) * partition_length,))
                    indices = np.clip(
                        indices, 0, partition_length - 1).astype(np.int64)

                    # Move indices to correct location within the video
                    indices = indices + clip_idx * partition_length

                # This means that partition overlap is allowed
                else:
                    # What is the actual length of the video?
                    sample_len = min(self.clip_len, len(vr)) - 1
                    # Sample the indices that fit into partition clip_idx
                    indices = np.linspace(
                        0, sample_len, num=sample_len // self.frame_step)
                    indices = np.concatenate((indices, np.ones(
                        self.frames_per_clip - sample_len // self.frame_step) * sample_len,))
                    # Only take frames that lie within the partition clip
                    indices = np.clip(indices, 0, sample_len - 1)

                    clip_step = 0
                    if len(vr) > self.clip_len:
                        clip_step = (
                            len(vr) - self.clip_len) // (self.num_clips - 1)
                    indices = indices + clip_idx * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        return vr.get_batch(all_indices), clip_indices
