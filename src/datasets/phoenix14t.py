import glob
import logging
import os
from typing import Any, Dict, Literal, Optional
import PIL
import lightning
import pandas as pd
import torch
import torchvision
import torch.utils
import torch.utils.data
import torchvision.transforms.functional

from datasets.clipable_video_dataset import ClipableVideoDataset


class Phoenix14T(lightning.LightningDataModule):

    def __init__(
        self,
        dataset: dict = dict(),
        train_dataset: dict = dict(),
        val_dataset: dict = dict(),
        test_dataset: dict = dict(),
        dataloader: dict = dict(),
        train_dataloader: dict = dict(),
        val_dataloader: dict = dict(),
        test_dataloader: dict = dict()
    ) -> None:
        super(Phoenix14T, self).__init__()

        self.train_dataset_cfg = dataset
        self.train_dataset_cfg.update(train_dataset)
        self.train_dataset_cfg['split'] = 'train'
        self.val_dataset_cfg = dataset
        self.val_dataset_cfg.update(val_dataset)
        self.val_dataset_cfg['split'] = 'dev'
        self.test_dataset_cfg = dataset
        self.test_dataset_cfg.update(test_dataset)
        self.test_dataset_cfg['split'] = 'test'

        self.train_dataloader_cfg = dataloader
        self.train_dataloader_cfg.update(train_dataloader)
        self.val_dataloader_cfg = dataloader
        self.val_dataloader_cfg.update(val_dataloader)
        self.test_dataloader_cfg = dataloader
        self.test_dataloader_cfg.update(test_dataloader)

        # TODO: This should go in the setup() method, but is not being called however
        self.train_dataset = Phoenix14TDataset(**self.train_dataset_cfg)
        self.val_dataset = Phoenix14TDataset(**self.val_dataset_cfg)
        self.test_dataset = Phoenix14TDataset(**self.test_dataset_cfg)

    def train_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(self.train_dataset, **self.train_dataloader_cfg)

    def val_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(self.val_dataset, **self.val_dataloader_cfg)

    def test_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(self.test_dataset, **self.test_dataloader_cfg)

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


class Phoenix14TDataset(ClipableVideoDataset):

    def __init__(
        self,
        root: str,
        split: Literal['train', 'train-complex-annotation', 'test', 'dev'],
        *,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.root = root
        self.split = split
        super(Phoenix14TDataset, self).__init__(min_len, max_len)

    def init_meta(self) -> pd.DataFrame:
        annotations_path = calc_annotations_path(self.root, self.split)
        meta_path = calc_meta_path(calc_meta_dir(self.root), self.split)
        annotations = read_annotations(annotations_path)
        meta = read_meta(meta_path)
        return meta.merge(annotations, how='inner', on='video_name')

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(
        self,
        index: int
    ) -> Dict[str, Any]:
        # TODO: Unstretch images according to README.md
        row = self.meta.iloc[index]
        sample = dict(frames=load_video(self.root, self.split, row.video_name))
        sample.update(row.to_dict())
        return sample


def load_video(root: str, split: Literal['train', 'test', 'dev', 'train-complex-annotations'], video_name: str) -> torch.Tensor:
    frame_paths = glob.glob(video_glob_pattern(root, split, video_name))
    frame_paths = sorted(frame_paths)
    frames = []
    for path in frame_paths:
        frame = PIL.Image.open(path)
        frame = torchvision.transforms.functional.pil_to_tensor(frame)
        frames.append(frame)
    frames = torch.stack(frames)
    return frames


def calc_annotations_path(root: str, split: str) -> str:
    return os.path.join(root, 'annotations', 'manual', f'PHOENIX-2014-T.{split}.corpus.csv')


def calc_meta_path(meta_dir: str, split: str) -> str:
    return os.path.join(meta_dir, f'{split}.csv')


def calc_meta_dir(root: str) -> str:
    return os.path.join(root, 'meta')


def read_annotations(path: str) -> pd.DataFrame:
    annot = pd.read_csv(
        filepath_or_buffer=path,
        sep='|',
        dtype=dict(
            name=str,
            video=str,
            start=int,
            end=int,
            speaker=str,
            orth=str,
            translation=str
        )
    )
    # This renaming is done, because the column name 'name' from the file clashes with how pandas indexes single DataFrame rows.
    annot.columns = ['video_name'] + annot.columns.to_list()[1:]
    return annot


def read_meta(path: str):
    return pd.read_csv(
        filepath_or_buffer=path,
        sep=';',
        dtype=dict(
            video_name=str,
            num_frames=int
        )
    )


def video_glob_pattern(root: str, split: Literal['train', 'dev', 'test', 'train-complex-annotations'], video_name: str) -> str:
    return os.path.join(root, 'features', 'fullFrame-210x260px', split, video_name, '*.png')


def create_meta(root: str, split: str, *, annotations: Optional[pd.DataFrame] = None, meta_path: Optional[str] = None, overwrite: bool = False) -> None:
    # TODO: This does not work for train-complex-annotations split. why?

    if meta_path is None:
        meta_dir = calc_meta_dir(root)
        meta_path = calc_meta_path(meta_dir, split)
    else:
        meta_dir = os.path.dirname(meta_path)
    if os.path.exists(meta_path) and not overwrite:
        return
    if annotations is None:
        annotations = read_annotations(calc_annotations_path(root, split))
    ds = []
    for _index, row in annotations.iterrows():
        video_path = video_glob_pattern(root, split, row.video_name)
        frame_paths = glob.glob(video_path)
        ds.append(
            dict(
                video_name=row.video_name,
                num_frames=len(frame_paths)
            )
        )
    meta = pd.DataFrame(ds)

    if not os.path.exists(meta_dir):
        os.mkdir(meta_dir)

    meta.to_csv(meta_path, sep=';', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=str)
    args = parser.parse_args()
    for split in ['train', 'dev', 'test']:
        create_meta(args.root, split, annotations=read_annotations(
            calc_annotations_path(args.root, split)))
