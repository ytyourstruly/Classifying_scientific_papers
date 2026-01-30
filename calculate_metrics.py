import argparse
import glob
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(csv_file, truth_col='truth', pred_col='prediction', pos_label='Human'):
    model_name = os.path.basename(csv_file)
    df = pd.read_csv(csv_file)

    if truth_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Expected columns '{truth_col}' and '{pred_col}' in {csv_file}")

    y_true = df[truth_col]
    y_pred = df[pred_col]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    return {
        'file': model_name,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for evaluation CSV(s)')
    parser.add_argument('--input', '-i', default=None,
                        help='Input CSV file, glob pattern, or directory containing CSVs (default: none)')
    parser.add_argument('--output', '-o', default=None, help='Optional output CSV to save aggregated metrics')
    parser.add_argument('--truth-col', default='truth', help='Column name for ground truth labels')
    parser.add_argument('--pred-col', default='prediction', help='Column name for predicted labels')
    parser.add_argument('--pos-label', default='Human', help='Positive label for binary metrics')

    args = parser.parse_args()

    if args.input is None:
        raise SystemExit('Please provide --input FILE or glob pattern (e.g. csv_files/*.csv)')

    # Resolve input files
    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, '*.csv'))
    else:
        files = glob.glob(args.input)

    results = []
    for f in files:
        try:
            res = calculate_metrics(f, truth_col=args.truth_col, pred_col=args.pred_col, pos_label=args.pos_label)
            results.append(res)
            print(f"{res['file']}: accuracy={res['accuracy']} precision={res['precision']} recall={res['recall']} f1={res['f1']}")
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if args.output and results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Saved aggregated metrics to {args.output}")


if __name__ == '__main__':
    main()
