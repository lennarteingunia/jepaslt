import argparse
from enum import Enum
from functools import reduce
import glob
import os
from typing import List, Tuple

import pandas as pd

from patterns import split_paths, ranked_confidences_pattern, ranked_confusion_matrix_pattern, ranked_votes_pattern


class CombinationType(Enum):
    ConfusionMatrix = 'confusion_matrix'
    Confidences = 'confidences'
    Votes = 'votes'


def combine_confusion_matrizes(paths: List[str]) -> List[pd.DataFrame]:

    paths = [(path, int(ranked_confusion_matrix_pattern.match(path).group('split')))
             for path in paths]
    splits = set(map(lambda x: x[1], paths))
    paths = [(x, [y[0] for y in paths if y[1] == x]) for x in splits]

    col = []
    for split, split_paths in paths:
        cms = [pd.read_csv(path, sep=' ', header=None).to_numpy()
               for path in split_paths]
        cm = reduce(lambda m1, m2: m1 + m2, cms)
        col.append((split, cm))
    return col


def combine_confidences_and_predict(confidence_csv_paths) -> List[Tuple[int, pd.DataFrame]]:

    def _combine_confidences_and_predict(paths) -> pd.DataFrame:
        dfs = [pd.read_csv(path, sep=' ', header=None) for path in paths]
        df = pd.concat(dfs)
        df.columns = ['path', *df.columns[1:-1], 'prediction']
        df = df.groupby(['path']).agg(
            {column: 'sum' for column in df.columns[1:-1]}).reset_index()
        df['prediction'] = df[df.columns[1:]].idxmax(axis=1) - 1
        return df

    paths = [(path, int(ranked_confidences_pattern.match(path).group('split')))
             for path in confidence_csv_paths]
    splits = set(map(lambda x: x[1], paths))
    paths = [(x, [y[0]for y in paths if y[1] == x]) for x in splits]

    return [(split, _combine_confidences_and_predict(path)) for (split, path) in paths]


def combine_votes_and_predict(vote_paths) -> List[Tuple[int, pd.DataFrame]]:

    def _combine_votes_and_predict(paths) -> pd.DataFrame:
        dfs = [pd.read_csv(path, sep=' ', header=None) for path in paths]
        df = pd.concat(dfs)
        df.columns = ['path', *df.columns[1:-1], 'prediction']
        df = df.groupby(['path']).agg(
            {column: 'sum' for column in df.columns[1:-1]}).reset_index()
        df['prediction'] = df[df.columns[1:]].idxmax(axis=1) - 1
        return df

    paths = [(path, int(ranked_votes_pattern.match(path).group('split')))
             for path in vote_paths]
    splits = set(map(lambda x: x[1], paths))
    paths = [(x, [y[0]for y in paths if y[1] == x]) for x in splits]

    return [(split, _combine_votes_and_predict(path)) for (split, path) in paths]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument(
        '--type', choices=[e.value for e in CombinationType], required=True)
    args = parser.parse_args()

    combination_type = CombinationType(args.type)

    csv_paths = glob.glob(os.path.join(args.path, '*.csv'))
    confidence_csv_paths, vote_csv_paths, confusion_matrix_paths = split_paths(
        csv_paths)

    # # Be careful as these combined confusion matrizes don't actually have any meaning (as we don't know which video was predicted when and where)

    # cms = combine_confusion_matrizes(confusion_matrix_paths)
    # for split, cm in cms:
    #     pd.DataFrame(cm).to_csv(os.path.join(
    #         args.path, f'{split}_confusion_matrix.csv'), header=False, index=False)

    # combined_confidences = combine_confidences_and_predict(
    #     confidence_csv_paths)

    confidence_based_votes = combine_confidences_and_predict(
        confidence_csv_paths)

    for split, votes in confidence_based_votes:
        path = os.path.join(args.path, f'{split}_confidences.csv')
        votes.to_csv(path, header=False, index=False, sep=' ')
        print(f'Saving confidences file to: {path}')

    direct_votes = combine_votes_and_predict(vote_csv_paths)

    for split, votes in direct_votes:
        path = os.path.join(args.path, f'{split}_votes.csv')
        votes.to_csv(path, header=False, index=False, sep=' ')
        print(f'Saving votes file to: {path}')
