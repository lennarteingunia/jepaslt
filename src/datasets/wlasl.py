import json
import os
from typing import Literal, Optional
import ffmpeg
import lightning
import pandas as pd
import tqdm

from datasets.video import VideoDataset


class WLASL(lightning.LightningDataModule):

    def __init__(
        self,
        root: str,
        frames_per_clip: int = 16,
        frame_step: int = 4,
        num_clips: int = 1,
        shared_transforms: list = list(),
        clip_transforms: list = list(),
        label_transforms: list = list(),
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,
        num_classes: Literal["2000", "1000", "300", "100"] = "2000",
        *,
        version: Literal["v0.3"] = "v0.3"
    ) -> None:
        super(WLASL, self).__init__()

        self.root = root
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.shared_transforms = shared_transforms
        self.clip_transforms = clip_transforms
        self.label_transforms = label_transforms
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.num_classes = int(num_classes)
        self.version = version

    def setup(self, stage: str) -> None:
        pass

    def prepare_data(self) -> None:
        WLASL.prepare_split(self.root, 'train', self.num_classes, self.version)
        WLASL.prepare_split(self.root, 'val', self.num_classes, self.version)
        WLASL.prepare_split(self.root, 'test', self.num_classes, self.version)

    def make_dataset(self, path) -> VideoDataset:
        return VideoDataset(
            path,
            frames_per_clip=self.frames_per_clip,
            frame_step=self.frame_step,
            num_clips=self.num_clips,
            shared_transforms=self.shared_transforms,
            clip_transforms=self.clip_transforms,
            label_transforms=self.label_transforms,
            random_clip_sampling=self.random_clip_sampling,
            allow_clip_overlap=self.allow_clip_overlap
        )

    @staticmethod
    def prepare_split(root: str, split: str, num_classes: int, version: str) -> None:
        print(f"Preparing the {split} split of the WLASL{num_classes} dataset...")
        print(f"We first prepare WLASL{num_classes} and then get smaller subsets of it")

        meta_path = os.path.join(root, 'metadata', f'{split}{num_classes}.csv')
        if os.path.exists(meta_path):
            return # This means the conversion was already done before
        old_meta_path = os.path.join(root, 'metadata', f"WLASL_{version}.json")
        with open(old_meta_path) as meta_file:
            meta = json.load(meta_file)

        meta_2000_path = os.path.join(root, 'metadata', f'{split}2000.csv')

        if not os.path.exists(meta_2000_path): # If we have no appropriate meta_file, we create the 2000 one and split it later.
            new_data = []
            for entry in tqdm.tqdm(meta, desc="glosses", position=0):
                gloss = entry["gloss"]
                for instance in tqdm.tqdm(entry["instances"], desc=f"{gloss}", position=1, leave=False):
                    if instance["split"] != split:
                        continue # This means the instance is not in the correct split

                    video_id = instance["video_id"]
                    video_path = os.path.join(root, "data", f"{video_id}.mp4")

                    if not os.path.exists(video_path):
                        continue # Skipping any video that does not actually exist. This can be caused when the dataset was not downloaded completely

                    
                    streams = ffmpeg.probe(video_path)["streams"]
                    if len(streams) != 1:
                        continue # Skipping this video because something is wrong with. 
                    video_length = streams[0]["nb_frames"] # This might rarely be wrong, because this is read from the header. To actually get the real number of frames, I would need to decode the whole video.

                    new_data.append(dict(
                        fname=video_path,
                        video_length=video_length,
                        speaker=instance["signer_id"],
                        glosses=gloss,
                        label=gloss # For this dataset, we make no difference between label and gloss annotation, because there is no linguistic information anyway.
                    ))
            
            new_data = pd.DataFrame(new_data)
            new_data.to_csv(meta_2000_path, sep=';')
        
        if num_classes != 2000: # No need to possibly do this twice
            meta_2000 = pd.read_csv(meta_2000_path, sep=';') # Now we have the meta file in correct structure but with possibly too many entries
            print(meta_2000)
                

if __name__ == "__main__":
    ds = WLASL(root="/mnt/datasets/wlasl/WLASL2000")
    ds.prepare_data()