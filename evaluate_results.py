from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from constants import *
from datasets import *
from models import *
from utils.model_utils import *
from utils.data_utils import get_roadmap_col_order, load_mpra_data
from utils.metrics import show_prec_recall, pr_summary


def concat_all_scores(eval_task, split='all', na_thresh=0.05):
    """ output score file contains chr, pos, label, nn_scores. For evaluation
    we want additional scores for regbase, eigen, etc. Concat these to the
    score file, excluding non-E116 roadmap scores.
    """
    proj_dir = PROCESSED_DIR / eval_task

    scores_pos_neg = pd.read_csv(
        proj_dir / 'output' / f'preds_trained_on_e116_pos_neg.csv', sep=',')
    scores_pos_bg = pd.read_csv(
        proj_dir / 'output' / f'preds_trained_on_e116_pos_bg.csv', sep=',')

    addl = pd.read_csv(proj_dir / f'matrix_{split}.csv', sep=',')
    assert all(scores_pos_neg.pos == addl.pos)
    assert all(scores_pos_bg.pos == addl.pos)

    scores_pos_bg.drop(['chr', 'pos', 'Label'], axis=1, inplace=True)
    addl.drop(['chr', 'pos', 'Label'], axis=1, inplace=True)
    omit_roadmap = [x for x in get_roadmap_col_order() if x[-3:] != '116']
    addl.drop(omit_roadmap, axis=1, inplace=True)

    # drop scores that have >5% NaNs from metrics (were dropped from nn as well)
    na_filt = (addl.isna().sum() > na_thresh * len(addl))
    omit_cols = addl.columns[na_filt].tolist()
    omit_cols += [x + '_PHRED' for x in omit_cols if x + '_PHRED' in addl.columns]
    addl.drop(omit_cols, axis=1, inplace=True)

    scores = pd.concat([scores_pos_neg, scores_pos_bg, addl], axis=1)
    return scores


def add_genonet_scores(project, scores):
    proj_dir = PROCESSED_DIR / project

    gn = pd.read_csv(proj_dir / 'test_genonet.tsv', sep='\t')
    gn = gn.loc[:, ['chr', 'pos_start', 'GenoNet-E116']]
    gn.rename(columns={'pos_start': 'pos'}, inplace=True)

    gn['chr'] = gn['chr'].map(lambda x: x[3:])
    gn[['chr', 'pos']] = gn[['chr', 'pos']].astype(int)
    scores = pd.merge(scores, gn, on=['chr', 'pos'], how='left')
    return scores


def score_metric_comparison(scores, metric='AUC'):
    if metric == 'AUC':
        scorer = lambda x: roc_auc_score(scores.Label[~x.isna()], x[~x.isna()])
    elif metric == 'APR':
        scorer = lambda x: average_precision_score(scores.Label[~x.isna()], x[~x.isna()])

    try:
        results = scores.drop(
            ['chr', 'pos', 'Label', 'Pool', 'pvalue_expr',
            'padj_expr', 'pvalue_allele', 'padj_allele'], axis=1).apply(scorer)
    except KeyError:
        results = scores.drop(['chr', 'pos', 'Label'], axis=1).apply(scorer)
    return results


if __name__ == '__main__':

    eval_out_dir = Path('./analysis')

    # ---------- Cross-eval both models on both test sets ----------- #
    for trained_project in ['e116_pos_bg', 'e116_pos_neg']:
        for to_eval in ['e116_pos_bg', 'e116_pos_neg']:
            print(f'Predicting: {trained_project} on {to_eval}')
            evl = Evaluator(trained_data=trained_project, eval_data=to_eval)
            evl.setup_data(model='neighbors', split='test')
            evl.predict_model()
            evl.save_scores()

    # ---------- Combine all model performances ------------ #
    for to_eval in ['e116_pos_bg', 'e116_pos_neg']:
        scores = concat_all_scores(to_eval, split='test')
        scores = add_genonet_scores(to_eval, scores)
        scores.index.names = ['scores']
        scores.to_csv(PROCESSED_DIR / to_eval / f'output/all_scores_{to_eval}.csv',
                    index=False)
    
        scores_clean = scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
        auc_scores = score_metric_comparison(scores_clean, 'AUC')
        auc_scores = auc_scores.to_frame(name='test_auc')

        aupr_scores = score_metric_comparison(scores_clean, 'APR')
        aupr_scores = aupr_scores.to_frame(name='test_aupr')

        tab = pd.merge(auc_scores, aupr_scores, left_index=True, right_index=True)
        tab.index.names = ['scores']
        tab.reset_index().to_csv(eval_out_dir / f'score_comparison_{to_eval}.csv', index=False)
        print(tab)


    ############ Precision analysis #################
    N_VAR = 9254535
    top_quants = list(np.linspace(0.00001, 0.001, 200)) + list(np.linspace(0.001, 0.055, 1000)) + \
        [25000 / N_VAR, 50000 / N_VAR, 75000 / N_VAR, 100000 / N_VAR]
    thresholds = [1 - x for x in top_quants]

    me = MultipleEval(
        ['1kg_background2', '1kg_background3', '1kg_background4', '1kg_background5'],
        trained_data='e116_pos_bg')
    # me.eval_all()

    score_recs = {}
    for SCORE in ['mpra_screen_trained_on_e116_pos_bg',
                  'DNase-E116', 'CADD_PHRED', 'DANN_PHRED', 'FATHMM-MKL_PHRED',
                  'FunSeq2_PHRED', 'GenoCanyon_PHRED', 'FIRE_PHRED', 'ReMM_PHRED',
                  'LINSIGHT_PHRED', 'fitCons_PHRED', 'FitCons2_PHRED']:
        print(SCORE)
        me.calibrate(score=SCORE)
        me.load_holdout_prediction('e116_pos_bg')
        recalls = me.estimate_recall(thresholds)
        score_recs[f'recall_{SCORE}'] = recalls

    df = pd.DataFrame(score_recs)
    df['top_N_frac'] = top_quants
    print(df)
    df.to_csv(eval_out_dir / 'recall_results_new.csv', index=False)
