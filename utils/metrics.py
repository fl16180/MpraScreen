import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def metric_report(truth, predictions):

    print('Metric report: ')
    print('-----------------------------------------------------------------')

    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    sens = 1. * tp / (tp + fn)
    spec = 1. * tn / (tn + fp)
    recall = 1. * tp / (tp + fn)
    precision = 1. * tp / (tp + fp)

    f1 = 2 * (precision * recall) / (precision + recall)

    print('sensitivity: {0} \t specificity: {1}'.format(sens, spec))
    print('precision: {0} \t recall: {1}'.format(precision, recall))

    print('F1 score: {0}'.format(f1))
    print('confusion matrix: \n', confusion_matrix(truth, predictions))
    print('-----------------------------------------------------------------\n')

    return tn, fp, fn, tp, f1


def show_prec_recall(y_true, probs):
    prec, rec, thresh = precision_recall_curve(y_true, probs)
    pr_df = pd.DataFrame({'precision': prec, 'recall': rec, 'threshold': np.append(0, thresh)})
    pr_df['n_above_threshold'] = pr_df['threshold'].map(lambda x: np.sum(probs > x))
    return pr_df


def pr_summary(y_true, probs, precs=[0.98, 0.95, 0.9, 0.85, 0.8, 0.75]):
    pr_df = show_prec_recall(y_true, probs)
    out = pd.DataFrame()
    for thr in precs:
        tmp = pr_df[pr_df['precision'] > thr]
        row = tmp[tmp['recall'] == max(tmp['recall'])].copy()

        row = row.sort_values(by='precision', ascending=False).iloc[:1, :]

        row['prec >= xx'] = thr
        out = pd.concat([out, row])
    out = out.loc[:, ['prec >= xx', 'recall', 'precision', 'threshold', 'n_above_threshold']]
    return out