# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class Meter(ABC):

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError(
            f'{Meter.reset.__name__} has not been implemented')

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError(
            f'{Meter.update.__name__} has not been implemented')


class DistributedMeter(Meter):

    def __init__(self) -> None:
        super(DistributedMeter, self).__init__()

    def reset(self) -> None:
        super(DistributedMeter, self).reset()


class AverageMeter(Meter):
    """computes and stores the average and current value"""

    def __init__(self):
        super(AverageMeter, self).__init__()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PerClassAverageMeter(Meter):

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        super(PerClassAverageMeter, self).__init__()

    @property
    def top_1_acc(self):
        return self.confusion_matrix.diagonal().sum() / self.confusion_matrix.sum()

    @property
    def per_class_top1_acc(self):
        return self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=1)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        for label, prediction in zip(labels.view(-1), predictions.view(-1)):
            self.confusion_matrix[label.long(), prediction.long()] += 1

    def _revert(self, predictions: torch.Tensor, labels: torch.Tensor):
        for label, prediction in zip(labels.view(-1), predictions.view(-1)):
            self.confusion_matrix[label.long(), prediction.long()] -= 1

    def to_file(
        self,
        dir: str,
        *,
        file_prefix: str = '',
        file_suffix: str = ''
    ) -> None:
        with open(os.path.join(dir, f'{file_prefix}confmat{file_suffix}.csv'), 'w') as f:
            df = pd.DataFrame(self.confusion_matrix)
            df.to_csv(f, sep=' ', header=False, index=False)


class PerClassConfidenceWeightedFullVideoPredictionMeter(PerClassAverageMeter):

    def __init__(
        self,
        num_classes: int
    ) -> None:
        super(PerClassConfidenceWeightedFullVideoPredictionMeter,
              self).__init__(num_classes=num_classes)

    def reset(self) -> None:
        super(PerClassConfidenceWeightedFullVideoPredictionMeter, self).reset()
        self.summed_clip_confidences = dict()

    def update(self, confidences: torch.Tensor, video_paths: list[str], labels: torch.Tensor):

        for confidence, path, label in zip(confidences, video_paths, labels):

            if not path in self.summed_clip_confidences:

                self.summed_clip_confidences[path] = np.zeros(
                    (self.num_classes,))

            else:

                self._revert(torch.tensor(
                    np.argmax(self.summed_clip_confidences[path])), torch.tensor(label))

            self.summed_clip_confidences[path] += confidence.detach().cpu().numpy()

            super(PerClassConfidenceWeightedFullVideoPredictionMeter, self).update(
                torch.tensor(np.argmax(self.summed_clip_confidences[path])), torch.tensor(label))

    def to_file(self, dir: str, *, file_prefix: str = '', file_suffix: str = '') -> None:

        with open(os.path.join(dir, f'{file_prefix}confidence_weighted_video_predictions{file_suffix}.csv'), 'w') as f:

            video_predictions = [dict(path=path, prediction=np.argmax(summed_clip_confidence))
                                 for path, summed_clip_confidence in self.summed_clip_confidences.items()]
            video_predictions_df = pd.DataFrame(video_predictions)
            video_predictions_df.to_csv(f, sep=' ', header=False, index=False)

        PerClassAverageMeter.to_file(
            self, dir, file_prefix=f'{file_prefix}confidence_weighted_video_predictions_', file_suffix=file_suffix)


class PerClassWeightedFullVideoPredictionMeter(PerClassAverageMeter):

    def __init__(self, num_classes: int) -> None:
        super(PerClassWeightedFullVideoPredictionMeter, self).__init__(
            num_classes=num_classes)

    def reset(self) -> None:
        PerClassAverageMeter.reset(self)
        self.summed_clip_predictions = dict()

    def update(self, predictions: torch.Tensor, video_paths: list[str], labels: torch.Tensor):

        for prediction, path, label in zip(predictions.view(-1), video_paths, labels.view(-1)):

            if not path in self.summed_clip_predictions:

                self.summed_clip_predictions[path] = np.zeros(
                    (self.num_classes,))

            else:

                self._revert(torch.tensor(
                    np.argmax(self.summed_clip_predictions[path])), torch.tensor(label))

            self.summed_clip_predictions[path][prediction] += 1

            PerClassAverageMeter.update(self, torch.tensor(
                np.argmax(self.summed_clip_predictions[path])), torch.tensor(label))

    def to_file(self, dir: str, *, file_prefix: str = '', file_suffix: str = '') -> None:

        with open(os.path.join(dir, f'{file_prefix}weighted_full_video_predictions{file_suffix}.csv'), 'w') as f:

            video_predictions = [dict(path=path, prediction=np.argmax(summed_clip_prediction))
                                 for path, summed_clip_prediction in self.summed_clip_predictions.items()]
            video_predictions_df = pd.DataFrame(video_predictions)
            video_predictions_df.to_csv(f, sep=' ', header=False, index=False)

        PerClassAverageMeter.to_file(
            self, dir, file_prefix=f'{file_prefix}weighted_video_predictions_', file_suffix=file_suffix)


class PerClassPredictionPositionMeter(PerClassAverageMeter):

    def __init__(
        self,
        num_classes: int,
        *,
        binning: bool = True,
        num_bins: Optional[int] = 100
    ) -> None:
        assert num_bins > 0

        self.binning = binning
        self.num_bins = num_bins

        super(PerClassPredictionPositionMeter, self).__init__(
            num_classes=num_classes)

    @property
    def per_class_average_correct_positions(self) -> np.ndarray:
        per_class_correct_summed_relative_positions = self.summed_relative_start_positions.diagonal()
        per_class_correct_position_counts = self.confusion_matrix.diagonal()
        return per_class_correct_summed_relative_positions / per_class_correct_position_counts

    @property
    def average_correct_position(self) -> np.ndarray:
        correct_summed_relative_positions = self.summed_relative_start_positions.diagonal().sum()
        correct_position_counts = self.confusion_matrix.diagonal().sum()
        return correct_summed_relative_positions / correct_position_counts

    @property
    def per_class_average_incorrect_positions(self) -> np.ndarray:
        per_class_incorrect_summed_relative_positions = self.summed_relative_start_positions.sum(
            axis=1) - self.summed_relative_start_positions.diagonal()
        per_class_incorrect_position_counts = self.confusion_matrix.sum(
            axis=1) - self.confusion_matrix.diagonal()
        return per_class_incorrect_summed_relative_positions / per_class_incorrect_position_counts

    @property
    def average_incorrect_position(self) -> np.ndarray:
        incorrect_summed_relative_positions = (self.summed_relative_start_positions.sum(
            axis=1) - self.summed_relative_start_positions.diagonal()).sum()
        incorrect_position_counts = (self.confusion_matrix.sum(
            axis=1) - self.confusion_matrix.diagonal()).sum()
        return incorrect_summed_relative_positions / incorrect_position_counts

    def reset(self) -> None:

        PerClassAverageMeter.reset(self)

        self.summed_relative_start_positions = np.zeros(
            (self.num_classes, self.num_classes))

        if self.binning:
            self.bins = np.zeros(
                (self.num_classes, self.num_classes, self.num_bins))

    def update(self, predictions: torch.Tensor, labels: torch.Tensor, positions: torch.Tensor, video_lengths: torch.Tensor) -> None:

        PerClassAverageMeter.update(
            self, predictions=predictions, labels=labels)

        for pred, lab, pos, vl in zip(predictions.view(-1), labels.view(-1), positions.view(-1), video_lengths.view(-1)):

            relative_start_position = pos / vl
            self.summed_relative_start_positions[lab.long(
            ), pred.long()] += relative_start_position

            if self.binning:

                bin_num = math.floor(
                    relative_start_position / (self.num_bins ** -1))
                self.bins[lab.long(), pred.long(), bin_num] += 1

    def to_file(
        self,
        dir: str,
        *,
        file_prefix: str = '',
        file_suffix: str = ''
    ) -> None:

        if not os.path.exists(dir):
            raise ValueError(
                f'{dir} does not exist. Create the logging directory first.')
        if os.path.isfile(dir):
            raise ValueError(f'{dir} is a file, not a directory.')

        with open(os.path.join(dir, f'{file_prefix}summed_average_start_positions{file_suffix}.csv'), 'w') as sasp_file:
            df = pd.DataFrame(self.summed_relative_start_positions)
            df.to_csv(sasp_file, header=False, index=False)

        if self.binning:
            with open(os.path.join(dir, f'{file_prefix}bins{file_suffix}.npy'), 'w') as bins_file:
                self.bins.tofile(bins_file)

        PerClassAverageMeter.to_file(
            self, dir, file_prefix=file_prefix, file_suffix=file_suffix)


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats


def adamw_logger(optimizer):
    """ logging magnitude of first and second momentum buffers in adamw """
    # TODO: assert that optimizer is instance of torch.optim.AdamW
    state = optimizer.state_dict().get('state')
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    for key in state:
        s = state.get(key)
        exp_avg_stats.update(float(s.get('exp_avg').abs().mean()))
        exp_avg_sq_stats.update(float(s.get('exp_avg_sq').abs().mean()))
    return {'exp_avg': exp_avg_stats, 'exp_avg_sq': exp_avg_sq_stats}
