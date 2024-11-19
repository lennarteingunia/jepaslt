import argparse

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.file, header=None, sep='\t')
    cm = df.to_numpy()
    print(cm)

    per_class_cms = []
    for cls in range(len(cm)):
        TP = cm[cls, cls]
        FP = cm[:, cls].sum() - TP
        FN = cm[cls, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        per_class_cms.append(np.array([[TP, FN], [FP, TN]]))
    per_class_cms = np.stack(per_class_cms)

    class_prevalence = per_class_cms[:, 0].sum(axis=1) / cm.sum()
    class_accuracy = (per_class_cms[:, 0, 0] +
                      per_class_cms[:, 1, 1]) / cm.sum()

    class_precision = per_class_cms[:, 0, 0] / \
        (per_class_cms[:, 0, 0] + per_class_cms[:, 1, 0])
    class_recall = per_class_cms[:, 0, 0] / \
        (per_class_cms[:, 1, 1] + per_class_cms[:, 0, 1])
    class_specificity = per_class_cms[:, 0, 0] / (per_class_cms[:, 0, 0] + per_class_cms[:, 0, 1])

    class_f1_score = 2 * class_precision * \
        class_recall / (class_precision + class_recall)
    
    class_balanced_accuracy = (class_recall  + class_specificity) / 2

    unweighted_average_recall = class_recall.mean()
    weighted_average_recall = (class_recall * class_prevalence).sum()

    macro_f1_score = (class_prevalence * class_f1_score).sum()
    print(class_balanced_accuracy)
