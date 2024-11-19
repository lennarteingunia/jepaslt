import argparse

import pandas as pd
import sklearn.metrics as metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--split-file', required=True, nargs='+')
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.input_file, sep=' ', header=None)

    split = [pd.read_csv(path, sep=' ', header=None)
             for path in args.split_file]
    split = pd.concat(split)

    predictions = predictions[[
        predictions.columns[0], predictions.columns[-1]]]
    predictions.columns = ['path', 'prediction']
    predictions['path'] = predictions['path'].astype(str)
    predictions['prediction'] = predictions['prediction'].astype(int)
    predictions.set_index('path')

    split.columns = ['path', 'ground_truth']
    split['path'] = split['path'].astype(str)
    split['ground_truth'] = split['ground_truth'].astype(int)
    split.set_index('path')

    assert len(predictions) == len(
        split), f"There are {len(split) - len(predictions)} predictions missing!"

    joined = pd.merge(predictions, split)
    confusion_matrix = metrics.confusion_matrix(
        joined['ground_truth'], joined['prediction'])

    df_cm = pd.DataFrame(confusion_matrix)
    df_cm.to_csv(args.output_file, header=None, index=None, sep=' ')
    print(f'Saved to {args.output_file}')
