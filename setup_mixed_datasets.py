import argparse
import os
import pandas as pd
import numpy as np
from constants import *

from datasets import load_data_set, load_neighbors_set
from utils.data_utils import split_train_test, extract_genonet


def load_all_background(bg_proj, split):
    background_data = load_data_set(
        bg_proj, split=split, datasets=['roadmap', 'eigen', 'regbase'],
        make_new=False
    )
    bg_neighbors = load_neighbors_set(
        bg_proj, split=split, tissue='E116'
    )
    return background_data, bg_neighbors


def prepare_all_e116_data():
    mpra_data = load_data_set(
        'mpra_e116', split='all', datasets=['roadmap', 'eigen', 'regbase'],
        make_new=False
    )
    mpra_neighbors = load_neighbors_set(
        'mpra_e116', split='all', tissue='E116'
    )
    
    mpra_pos = mpra_data[mpra_data.Label == 1]
    neigh_pos = mpra_neighbors[mpra_data.Label == 1]
    mpra_original_neg = mpra_data[mpra_data.Label == 0]
    neigh_original_neg = mpra_neighbors[mpra_data.Label == 0]
    return mpra_pos, neigh_pos, mpra_original_neg, neigh_original_neg


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


def prepare_all_nova_data(bg_cols, fdr_thresh=0.05):
    mpra_data = load_data_set(
        'mpra_nova', split='all', datasets=['roadmap', 'eigen', 'regbase'],
        make_new=False
    )
    mpra_neighbors = load_neighbors_set(
        'mpra_nova', split='all', tissue='E116'
    )

    assert fdr_thresh > 0 and fdr_thresh < 1
    thresh = fdr_thresh    # use FDR 0.01 intersection

    # use index to match duplications/reorderings for neighbor data
    mpra_data['INDEX'] = np.arange(len(mpra_data))
    mpra_data = merge_with_validation_info(mpra_data, 'mpra_nova')
    mpra_neighbors = mpra_neighbors[mpra_data['INDEX'].values, :]
    
    # define functional as intersection of expression and allelic
    expr_sig = mpra_data['padj_expr'] < thresh
    allele_sig = mpra_data['padj_allele'] < thresh
    mpra_data['Label'] = np.logical_and(expr_sig, allele_sig).astype(int)

    # get duplication mask and drop duplicates
    dupls = mpra_data.duplicated(subset=['chr', 'pos', 'Label'])
    mpra_data = mpra_data.loc[~dupls, :]
    mpra_neighbors = mpra_neighbors[~dupls, :]

    # for mpra negatives, we need to take out chr/pos combos that are in the pos set from the negatives
    # since dropping duplicated won't take care of that case
    neg_fdr_thresh = 0.5
    print('Defining nova-seq pos/neg: ')
    print('\tTotal entries: ', mpra_data.shape[0])
    print('\tPos FDR thresh: ', fdr_thresh)
    print('\tNeg FDR thresh: ', neg_fdr_thresh)

    keep = (mpra_data.Label == 1) | ( (mpra_data.padj_expr > neg_fdr_thresh) & (mpra_data.padj_allele > neg_fdr_thresh))
    mpra_data = mpra_data.loc[keep, :]
    mpra_neighbors = mpra_neighbors[keep, :]
    print('\tRemaining entries: ', mpra_data.shape[0])

    print('\tExpr sig: ', expr_sig.sum(), 'Allele sig: ', allele_sig.sum())
    print('\tIntersection: ', mpra_data['Label'].sum())

    mpra_data.drop(['Pool', 'pvalue_expr', 'padj_expr', 'pvalue_allele',
                    'padj_allele'], axis=1, inplace=True)
    mpra_data = mpra_data.loc[:, bg_cols]
    
    mpra_pos = mpra_data[mpra_data.Label == 1]
    neigh_pos = mpra_neighbors[mpra_data.Label == 1]
    mpra_original_neg = mpra_data[mpra_data.Label == 0]
    neigh_original_neg = mpra_neighbors[mpra_data.Label == 0]
    return mpra_pos, neigh_pos, mpra_original_neg, neigh_original_neg


def sample_mask(length, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.choice(length, n, replace=False)
    mask = np.zeros(length)
    mask[idx] = 1
    return mask.astype(bool)


def subsample_neg(pos, neg, k=10):
    n_pos = pos.shape[0]
    Idx = np.random.permutation(neg.shape[0])[:int(k * n_pos)]
    return Idx


def arrange_background_bed():
    ''' The matched backgrounds for each possible relevant MPRA site are
        in a matrix data structure. We rearrange the data into bed format
        and give each background variant an associated tag.
    ''' 
    BG_MATCH = PROCESSED_DIR / 'matched_background/all_pos_background_set_hg19.csv'

    df = pd.read_csv(BG_MATCH)
    match_cols = [f'1-{x}' for x in range(15)]
    df.columns = ['drop', 'idx'] + match_cols
    df = df.drop('drop', axis=1)

    df[match_cols] = df[match_cols].applymap(lambda x: x.split('|')[0])

    df = pd.wide_to_long(
        df, stubnames=['1'], i=['idx'], j='match', sep='-'
    ).reset_index()

    df = df.rename(columns={'match': 'rs', '1': 'target'})
    df['rs'] = df['idx'].map(str) + ';' + df['rs'].map(str)
    df[['chr', 'pos']] = df['target'].str.split(':', expand=True)

    df[['chr_reord', 'pos_reord']] = df['idx'].str.split(':', expand=True).astype(int)
    df = df.sort_values(['chr_reord', 'pos_reord'], axis=0)
    df = df.drop(['idx', 'target', 'chr_reord', 'pos_reord'], axis=1)

    df['chr'] = df['chr'].map(lambda x: x.split('_')[0])
    chr_filter = [str(x) for x in range(1, 23)]
    df = df[df['chr'].isin(chr_filter)]

    df[['chr', 'pos']] = df[['chr', 'pos']].astype(int)
    df['pos_end'] = df['pos'] + 1
    df = df[['chr', 'pos', 'pos_end', 'rs']]

    df = df.reset_index(drop=True)
    df.to_csv(PROCESSED_DIR / 'matched_background/matched_background.bed', sep='\t', index=False, header=False)
    return df


def setup_mpra_pos_neg(args):
    ''' Sets up the MPRA positive vs inactive task.
    '''
    # ================= Load MPRA E116 positives ================== #
    print('Loading MPRA E116 positives')
    e116_pos, e116_neigh_pos, e116_neg, e116_neigh_neg = \
        prepare_all_e116_data()

    nova_pos, nova_neigh_pos, nova_neg, nova_neigh_neg = \
        prepare_all_nova_data(e116_pos.columns, fdr_thresh=0.01)

    print('DATASET INFO: ')
    print('\ttewhey pos/neg: ', e116_pos.shape[0], e116_neg.shape[0])
    print('\tnova-seq pos/neg: ', nova_pos.shape[0], nova_neg.shape[0])

    mpra_pos = pd.concat([e116_pos, nova_pos], axis=0)
    all_neigh_pos = np.vstack([e116_neigh_pos, nova_neigh_pos])

    # ============== Split positives into train/test =============== #
    print('Splitting train/test')
    train_mask = sample_mask(
        mpra_pos.shape[0], int(mpra_pos.shape[0] * 0.80), seed=0)
    test_mask = (1 - train_mask).astype(bool)

    train_pos = mpra_pos.iloc[train_mask]
    train_neigh_pos = all_neigh_pos[train_mask]

    test_pos = mpra_pos.iloc[test_mask]
    test_neigh_pos = all_neigh_pos[test_mask]
    print(train_pos.shape[0], test_pos.shape[0])

    # ============= Sample/split negatives from datasets ============ #
    print('Sampling negatives from datasets')
    idx = subsample_neg(e116_pos, e116_neg, k=args.match_k)
    e116_neg = e116_neg.iloc[idx, :]
    e116_neigh_neg = e116_neigh_neg[idx]

    idx = subsample_neg(nova_pos, nova_neg, k=args.match_k)
    nova_neg = nova_neg.iloc[idx, :]
    nova_neigh_neg = nova_neigh_neg[idx]

    mpra_neg = pd.concat([e116_neg, nova_neg], axis=0)
    all_neigh_neg = np.vstack([e116_neigh_neg, nova_neigh_neg])

    train_mask = sample_mask(
        mpra_neg.shape[0], int(mpra_neg.shape[0] * 0.85), seed=0)
    test_mask = (1 - train_mask).astype(bool)

    train_neg = mpra_neg.iloc[train_mask]
    train_neigh_neg = all_neigh_neg[train_mask]

    test_neg = mpra_neg.iloc[test_mask]
    test_neigh_neg = all_neigh_neg[test_mask]

    # ============== Concat pos and neg for train/test ============== #
    print('Merging positives and negatives')
    train_all = pd.concat([train_pos, train_neg], axis=0)
    train_neigh_all = np.vstack([train_neigh_pos, train_neigh_neg])
    test_all = pd.concat([test_pos, test_neg], axis=0)
    test_neigh_all = np.vstack([test_neigh_pos, test_neigh_neg])

    print(train_all.shape, train_neigh_all.shape)
    print(test_all.shape, test_neigh_all.shape)

    # ================== Saving ================ #
    print('Saving')
    train_all.to_csv(
        PROCESSED_DIR / 'e116_pos_neg' / 'matrix_train.csv', index=False)
    test_all.to_csv(
        PROCESSED_DIR / 'e116_pos_neg' / 'matrix_test.csv', index=False)
    np.save(
        PROCESSED_DIR / 'e116_pos_neg/neighbors/train_40_25_E116', train_neigh_all)
    np.save(
        PROCESSED_DIR / 'e116_pos_neg/neighbors/test_40_25_E116', test_neigh_all)


def setup_mpra_pos_bg(args):
    ''' Sets up the MPRA positive vs matched background task.

        Ensures that the same positives are split to train and test sets as
        in the positive vs inactive task
    '''
    bg_proj = 'matched_background'

    def make_lookup():
        ''' Create lookup dataframe from chr-pos to matches '''
        df = pd.read_csv(PROCESSED_DIR / bg_proj / 'all_pos_background_set_hg19.csv')
        match_cols = [f'1-{x}' for x in range(15)]
        df.columns = ['drop', 'idx'] + match_cols
        df = df.drop('drop', axis=1)

        df[['chr', 'pos']] = df['idx'].str.split(':', expand=True)
        df[['chr', 'pos']] = df[['chr', 'pos']].astype(int)

        df = df.drop('idx', axis=1)
        df[match_cols] = df[match_cols].applymap(lambda x: x.split('|')[0])
        return df


    def query_lookup(chr_pos, match_lookup, match_k=10):
        ''' Query lookup dataframe '''
        chr_pos = chr_pos[['chr', 'pos']]
        matches = chr_pos.merge(match_lookup, on=['chr', 'pos'])

        chr_match = []
        pos_match = []
        for row in matches.iterrows():
            for i in range(match_k):
                match_i = row[1][f'1-{i}']
                if match_i == 'NaN':
                    continue

                chr_i, pos_i = match_i.split(':')
                chr_i = chr_i.split('_')[0]
                if chr_i not in [str(x) for x in range(1, 23)]:
                    continue

                chr_match.append(int(chr_i))
                pos_match.append(int(pos_i))

        match_chr_pos = pd.DataFrame({'chr': chr_match, 'pos': pos_match})
        return match_chr_pos

    # ================= Load MPRA E116 positives ================== #
    print('Loading MPRA E116 positives')
    e116_pos, e116_neigh_pos, e116_neg, e116_neigh_neg = \
        prepare_all_e116_data()

    nova_pos, nova_neigh_pos, nova_neg, nova_neigh_neg = \
        prepare_all_nova_data(e116_pos.columns, fdr_thresh=0.01)

    print('DATASET INFO: ')
    print('\ttewhey pos/neg: ', e116_pos.shape[0], e116_neg.shape[0])
    print('\tnova-seq pos/neg: ', nova_pos.shape[0], nova_neg.shape[0])

    mpra_pos = pd.concat([e116_pos, nova_pos], axis=0)
    all_neigh_pos = np.vstack([e116_neigh_pos, nova_neigh_pos])

    # ================== Split into train/test ================== #
    print('Splitting train/test')
    train_mask = sample_mask(
        mpra_pos.shape[0], int(mpra_pos.shape[0] * 0.85), seed=0)
    test_mask = (1 - train_mask).astype(bool)

    train_pos = mpra_pos.iloc[train_mask]
    train_neigh_pos = all_neigh_pos[train_mask]

    test_pos = mpra_pos.iloc[test_mask]
    test_neigh_pos = all_neigh_pos[test_mask]

    train_chr_pos = train_pos[['chr', 'pos']]
    test_chr_pos = test_pos[['chr', 'pos']]
    print(train_chr_pos.shape[0], test_chr_pos.shape[0])

    # ================== Lookup background matches ================ #
    print('Lookup background matches')
    lookup = make_lookup()

    train_match = query_lookup(train_chr_pos, lookup)
    test_match = query_lookup(test_chr_pos, lookup)

    # ================== Merge with background data =============== #
    print('Merging with background data')
    background_data, bg_neighbors = load_all_background(bg_proj, 'all')

    indexer = background_data[['chr', 'pos']].copy()
    indexer['idx'] = np.arange(indexer.shape[0])
    train_matches = pd.merge(train_match, indexer, on=['chr', 'pos'])['idx'].values
    test_matches = pd.merge(test_match, indexer, on=['chr', 'pos'])['idx'].values

    train_bg = background_data.iloc[train_matches]
    train_bg_neighbors = bg_neighbors[train_matches]
    test_bg = background_data.iloc[test_matches]
    test_bg_neighbors = bg_neighbors[test_matches]

    train_all = pd.concat([train_pos, train_bg], axis=0)
    train_neigh_all = np.vstack([train_neigh_pos, train_bg_neighbors])
    test_all = pd.concat([test_pos, test_bg], axis=0)
    test_neigh_all = np.vstack([test_neigh_pos, test_bg_neighbors])
    print(
        train_all.shape[0], train_neigh_all.shape[0],
        test_all.shape[0], test_neigh_all.shape[0]
    )

    # ================== Saving ================ #
    print('Saving')
    train_all.to_csv(
        PROCESSED_DIR / 'e116_pos_bg' / 'matrix_train.csv', index=False)
    test_all.to_csv(
        PROCESSED_DIR / 'e116_pos_bg' / 'matrix_test.csv', index=False)
    np.save(
        PROCESSED_DIR / 'e116_pos_bg/neighbors/train_40_25_E116', train_neigh_all)
    np.save(
        PROCESSED_DIR / 'e116_pos_bg/neighbors/test_40_25_E116', test_neigh_all)


def setup_genonet(args):
    ''' Also download genonet scores for project test sets
    '''
    for proj in ['e116_pos_neg', 'e116_pos_bg']:
        test_data = pd.read_csv(PROCESSED_DIR / proj / 'matrix_test.csv')

        bed = test_data[['chr', 'pos']].copy()    
        bed['pos_end'] = bed['pos'].astype(int) + 1
        bed['rs'] = [f'test_{x}' for x in range(bed.shape[0])]
        bed = bed[['chr', 'pos', 'pos_end', 'rs']]

        extract_genonet(bed, PROCESSED_DIR / proj / 'test_genonet.tsv', 'mean')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arrange_bg', action='store_true')
    parser.add_argument('--pos_neg_task', action='store_true')
    parser.add_argument('--pos_bg_task', action='store_true')
    parser.add_argument('--match_k', type=int, default=10)
    parser.add_argument('--get_genonet', action='store_true')
    args = parser.parse_args()

    if args.arrange_bg:
        arrange_background_bed()

    if args.pos_neg_task:
        setup_mpra_pos_neg(args)

    if args.pos_bg_task:
        setup_mpra_pos_bg(args)

    if args.get_genonet:
        setup_genonet(args)
