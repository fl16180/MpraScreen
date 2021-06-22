import argparse
import os
import numpy as np
import pandas as pd

from constants import *
from datasets import load_data_set, load_neighbors_set
from utils.data_utils import split_train_test


def get_e116_pos():
    mpra_data = load_data_set(
        'mpra_e116', split='all', datasets=['roadmap', 'eigen', 'regbase'],
        make_new=False
    )
    mpra_neighbors = load_neighbors_set(
        'mpra_e116', split='all', tissue='E116'
    )
    
    mpra_pos = mpra_data[mpra_data.Label == 1]
    neigh_pos = mpra_neighbors[mpra_data.Label == 1]
    print('E116 positives: ', mpra_pos.shape[0])
    return mpra_pos, neigh_pos


def get_nova_pos():
    mpra_data = load_data_set(
        'mpra_nova', split='all', datasets=['roadmap', 'eigen', 'regbase'],
        make_new=False
    )
    mpra_neighbors = load_neighbors_set(
        'mpra_nova', split='all', tissue='E116'
    )
    thresh = 0.05    # use FDR 0.05 intersection

    # use index to match duplications/reorderings for neighbor data
    mpra_data['INDEX'] = np.arange(len(mpra_data))
    mpra_data = merge_with_validation_info(mpra_data, 'mpra_nova')
    mpra_neighbors = mpra_neighbors[mpra_data['INDEX'].values, :]
    
    # define functional as intersection of expression and allelic
    expr_sig = mpra_data['padj_expr'] < thresh
    allele_sig = mpra_data['padj_allele'] < thresh
    mpra_data['Label'] = np.logical_and(expr_sig, allele_sig).astype(int)

    # # get duplication mask and drop duplicates
    # dupls = mpra_data.duplicated(subset=['chr', 'pos', 'Label'])
    # mpra_data = mpra_data.loc[~dupls, :]
    # mpra_neighbors = mpra_neighbors[~dupls, :]

    print('NovaSeq2 positives:', mpra_data['Label'].sum())
    print('expr pos: ', expr_sig.sum())
    print('allelic pos: ', allele_sig.sum())

    mpra_data.drop(['Pool', 'pvalue_expr', 'padj_expr', 'pvalue_allele',
                    'padj_allele'], axis=1, inplace=True)
    
    mpra_pos = mpra_data[mpra_data.Label == 1]
    neigh_pos = mpra_neighbors[mpra_data.Label == 1]
    return mpra_pos, neigh_pos


def merge_with_validation_info(scores, eval_proj):
    """ Load full mpra_nova dataset in order to get full set of labels,
    i.e. the expression and allelic p-values.
    
    Join with scores dataframe using chr and pos columns.
    """
    ext_val_path = MPRA_DIR / MPRA_TABLE[eval_proj][0]
    dat = pd.read_csv(ext_val_path, sep='\t')
    dat.rename(columns={'chrom': 'chr', 'Hit': 'Label'}, inplace=True)
    dat = dat.loc[:,
        ['chr', 'pos', 'Pool', 'pvalue_expr',
        'padj_expr', 'pvalue_allele', 'padj_allele', 'Label']
    ]
    dat['chr'] = dat['chr'].map(lambda x: x[3:])
    dat[['chr', 'pos']] = dat[['chr', 'pos']].astype(int)

    scores.drop('Label', axis=1, inplace=True)
    scores = pd.merge(dat, scores, on=['chr', 'pos'])
    return scores


def add_matched_variants(bed):
    return

if __name__ == '__main__':
    
    save_files = False

    e116_pos, e116_neigh_pos = get_e116_pos()
    nova_pos, nova_neigh_pos = get_nova_pos()

    used_cols = e116_pos.columns.tolist()
    nova_pos = nova_pos.loc[:, used_cols].copy()

    all_pos = pd.concat([e116_pos, nova_pos], axis=0)
    all_pos = all_pos.loc[:, ['chr', 'pos', 'Label']]


    train_pos, test_pos = split_train_test(all_pos, test_frac=0.15, seed=0)

    if save_files:
        all_pos.to_csv(PROCESSED_DIR / 'gnom_mpra_mixed' / 'all_pos2.csv', index=False, header=False, sep=',')
        train_pos.to_csv(PROCESSED_DIR / 'gnom_mpra_mixed' / 'train_pos2.csv', index=False, header=False, sep=',')
        test_pos.to_csv(PROCESSED_DIR / 'gnom_mpra_mixed' / 'test_pos2.csv', index=False, header=False, sep=',')

