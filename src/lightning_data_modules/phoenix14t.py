# README:
#
# ===========================================================================================
# RWTH-PHOENIX-Weather 2014 T:  Parallel Corpus of Sign Language Video, Gloss and Translation
# ===========================================================================================


# This archive contains the RWTH-PHOENIX-Weather 2014 T corpus. It contains a multisigner set with matching gloss and translation sentence pairs.

# If you use this data in your research, please cite:

# Necati Cihan Camgöz, Simon Hadfield, Oscar Koller, Hermann Ney, Richard Bowden, Neural Sign Language Translation, IEEE Conf. on Computer Vision and Pattern
# Recognition, Salt Lake City, UT, 2018.

# ========================
# This corpus allows to:
# =========================

# - Use models trained on this RWTH Phoenix 2014 T corpus to ...

#   * evaluate against the baseline provided in the above mentioned paper

#   * evaluate on the RWTH-PHOENIX-Weather 2014 corpus, https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/, (see O. Koller, J. Forster, and H. Ney. Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. Computer Vision and Image Understanding, volume 141, pages 108-125, December 2015.)

#   * evaluate on the 1 Million Hands Articulated Handshape Challenge, https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/   (see O. Koller, H. Ney, and R. Bowden. Deep Hand: How to Train a CNN on 1 Million Hand Images When Your Data Is Continuous and Weakly Labelled. In CVPR 2016)

#   * evaluate on the Continuous Sign Language Mouthing Challenge, ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-phoenix/2016/phoenix-mouthing-ECCV.tar.gz (see O. Koller, H. Ney, and R. Bowden. Read My Lips: Continuous Signer Independent Weakly Supervised Viseme Recognition. In Proceedings of the 13th European Conference on Computer Vision (ECCV 2014), pages 281-296, Zurich, Switzerland, September 2014. )


# - Use models trained on RWTH-PHOENIX-Weather 2014 to evaluate on this RWTH-PHOENIX-Weather 2014 T

# ======================================
# It has been taken care to assure that:
# ======================================

# - No test, no dev segments from RWTH-PHOENIX-Weather 2014 will be found in this RWTH-PHOENIX-Weather 2014 T train set
# - No test, no dev segments from this RWTH-PHOENIX-Weather 2014 T corpus are present in train of RWTH-PHOENIX-Weather 2014
# - No segments that have been annotated for mouthing sequences in O. Koller, H. Ney, and R. Bowden. Read My Lips: Continuous Signer Independent Weakly Supervised Viseme Recognition. In ECCV 2014 are present in any sets


# ======================
# Additional information
# ======================

# - The segmentation comes mostly from  RWTH Phoenix 2014 (whereever the segment boundary times matched)
# - Where the segment boundaries did not match, we used a training alignment (split 9-2-0) with 200pca-reduced-1-million-hand-features (see O. Koller, H. Ney, and R. Bowden. Deep Hand: How to Train a CNN on 1 Million Hand Images When Your Data Is Continuous and Weakly Labelled. In CVPR 2016) to estimate the boundaries.
# - Therefore, some boundaries will not be correct
# - All segments originally containing "-(falsch: )" or "<??:GLOSS>" have been removed from dev and test. These usually represent signing errors and will therefore be problematic for end to end translation (video to german)

# ========
# Content
# ========

# The corpus contains 9 signers and has been recorded on the broadcastnews channel.

# PHOENIX-2014-T
# ├── annotations
# │   │
# │   └─ manual -> this contains the corpus files. Note that phoenix2016.train-complex-annotation.corpus.csv contains a more complex annotation. The evaluation scripts map this annotation back to the test protocoll
# │
# ├── evaluation -> this contains evaluation scripts for recognition and translation. WER is calculated with the Sclite tools. ROUGE & BLEU Scores are calculated by help of scripts from google.
# │
# ├── features
# │   │
# │   └─── fullFrame-210x260px -> resolution of 210x260 pixels, but they are distorted due to transmission channel particularities, to undistort stretch images to 210x300
# │      ├── dev
# │      ├── test
# │      └── train
# │ 
# └── models
#     │
#     └─── languageModels


import glob
import os
import lightning
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import pandas as pd
import ffmpeg
import torch
import torch.utils
import torch.utils.data
import tqdm

from datasets.full_video_dataset import FullVideoDataset


class Phoenix14T(lightning.LightningDataModule):

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
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 8,
    ) -> None:
        super(Phoenix14T, self).__init__()

        self.root = root
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.shared_transforms = shared_transforms
        self.clip_transforms = clip_transforms
        self.label_transforms = label_transforms
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    def setup(self, stage: str) -> None:
        meta_root = os.path.join(self.root, 'annotations', 'manual')
        if stage == "fit":
            self.train_ds = self.make_dataset(
                os.path.join(meta_root, 'train.csv'))
            self.val_ds = self.make_dataset(os.path.join(meta_root, 'dev.csv'))
        if stage == "test":
            self.test_ds = self.make_dataset(
                os.path.join(meta_root, 'test.csv'))
        if stage == "predict":
            raise NotImplementedError()

    def prepare_data(self) -> None:
        Phoenix14T.prepare_split(self.root, 'train')
        Phoenix14T.prepare_split(self.root, 'dev')
        Phoenix14T.prepare_split(self.root, 'test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def make_dataset(self, path) -> FullVideoDataset:
        return FullVideoDataset(
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
    def prepare_split(root: str, split: str) -> None:
        print(f"Preparing the {split} split of the Phoenix14T dataset...")
        meta_path = os.path.join(root, 'annotations', 'manual', f'{split}.csv')
        if os.path.exists(meta_path):
            return # This means we already prepared this once.
        old_meta = os.path.join(root, 'annotations', 'manual',
                                f'PHOENIX-2014-T.{split}.corpus.csv')
        data = pd.read_csv(old_meta, sep='|', dtype=dict(
            name=str,
            video=str,
            start=int,
            end=int,
            speaker=str,
            orth=str,
            translation=str,
        ))
        video_dir = os.path.join(root, 'videos', split)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        new_data = []
        for _index, row in tqdm.tqdm(data.iterrows(), total=len(data)):
            frame_pattern = os.path.join(
                root, 'features', 'fullFrame-210x260px', split, row["name"], '*.png')
            output_path = os.path.join(video_dir, f'{row["name"]}.mp4')

            if not os.path.exists(output_path):

                # Globbing all images and then outputting them in the resized size. -> NO resizing is needed as a transform
                (
                    ffmpeg
                    .input(frame_pattern, pattern_type='glob')
                    .filter('deflicker', mode='pm', size=10)
                    .filter('scale', width='-1', height='300')
                    .output(output_path, crf=20, preset='slower', movflags='faststart', pix_fmt='yuv420p', loglevel='quiet')
                    .run()
                )

            if os.path.exists(output_path):
                # This might seem confusing, but guarantees, that only videos that have been converted are indexed.
                # Make a new meta entry
                new_data.append(dict(
                    fname=output_path,
                    video_length=len(glob.glob(frame_pattern)),
                    speaker=row.speaker,
                    glosses=row.orth,
                    label=row.translation,
                ))

        new_meta = pd.DataFrame(new_data)
        new_meta.to_csv(meta_path, sep=';')
