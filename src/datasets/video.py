from typing import Any, List
import numpy as np
import pandas as pd
import torch
import decord
import tqdm


class FullVideoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_paths: List[str],
        frames_per_clip: int = 16,
        frame_step: int = 4,
    ) -> None:
        super(FullVideoDataset, self).__init__()

        video_paths, labels = [], []
        for data_path in data_paths:
            data = pd.read_csv(data_path, header=None, delimiter=' ')
            video_paths += list(data.values[:, 0])
            labels += list(data.values[:, 1])

        self.clip_len = int(frames_per_clip * frame_step)

        self.samples = []
        for video_path, label in tqdm.tqdm(zip(video_paths, labels), total=len(video_paths), desc='Indexing clips...'):
            self.video_reader = video_path, decord.VideoReader(
                video_path, num_threads=-1, ctx=decord.cpu(0))
            for start_idx in range(len(self.video_reader[1]) - self.clip_len):
                end_idx = start_idx + self.clip_len
                indices = np.linspace(start_idx, end_idx, num=frames_per_clip)
                indices = np.clip(indices, start_idx,
                                  end_idx - 1).astype(np.int64)
                self.samples.append({
                    'path': video_path,
                    'indices': indices,
                    'label': label
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> Any:

        sample = self.samples[index]

        if sample['path'] != self.video_reader[0]:
            self.video_reader = sample['path'], decord.VideoReader(
                sample['path'], num_threads=-1, ctx=decord.cpu(0))

        return self.video_reader[1].get_batch(sample['indices']), sample['label'], {'path': sample['path']}


if __name__ == '__main__':
    dataset = FullVideoDataset(
        ['/mnt/datasets/CREMA-D/additional/splits/split_0.csv']
    )
    print(f'Dataset Length is: {len(dataset)}')
    for clip, label in tqdm.tqdm(dataset):
        pass
