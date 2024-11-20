import argparse

import numpy as np
import pandas as pd


def calculate_metrics(cm: np.ndarray) -> None:
    per_class_cms = []
    for cls in range(len(cm)):
        TP = cm[cls, cls]
        FP = cm[:, cls].sum() - TP
        FN = cm[cls, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        per_class_cms.append(np.array([[TP, FP], [FN, TN]]))
    per_class_cms = np.stack(per_class_cms)
    accuracy = per_class_cms[:, 0, 0].sum() / cm.sum()

    class_prevalence = per_class_cms[:, 0].sum(axis=1) / cm.sum()
    class_accuracy = (per_class_cms[:, 0, 0] +
                      per_class_cms[:, 1, 1]) / cm.sum()

    class_precision = per_class_cms[:, 0, 0] / \
        (per_class_cms[:, 0, 0] + per_class_cms[:, 1, 0])
    class_recall = per_class_cms[:, 0, 0] / per_class_cms[:, 0, :].sum(axis=1)

    class_f1_score = 2 * class_precision * \
        class_recall / (class_precision + class_recall)

    macro_f1_score = class_f1_score.mean()
    weighted_f1_score = (class_f1_score * class_prevalence).sum()

    unweighted_average_recall = class_recall.mean()
    weighted_average_recall = (class_recall * class_prevalence).sum()

    return dict(
        accuracy=accuracy.item(),
        per_class_cms=per_class_cms.tolist(),
        class_prevalence=class_prevalence.tolist(),
        class_accuracy=class_accuracy.tolist(),
        class_precision=class_precision.tolist(),
        class_recall=class_recall.tolist(),
        class_f1_score=class_f1_score.tolist(),
        macro_f1_score=macro_f1_score.item(),
        weighted_f1_score=weighted_f1_score.item(),
        unweighted_average_recall=unweighted_average_recall.item(),
        weighted_average_recall=weighted_average_recall.item()
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.file, header=None, sep='\t')

    cm = df.to_numpy()

    calculate_metrics(df)