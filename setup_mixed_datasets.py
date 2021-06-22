import argparse
import os
import pandas as pd
import numpy as np
from constants import *

from datasets import load_data_set, load_neighbors_set
from utils.data_utils import split_train_test


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


def prepare_all_nova_data(bg_cols):
    mpra_data = load_data_set(
        'mpra_nova', split='all', datasets=['roadmap', 'eigen', 'regbase'],
        make_new=False
    )
    mpra_neighbors = load_neighbors_set(
        'mpra_nova', split='all', tissue='E116'
    )
    thresh = 0.05    # use FDR 0.01 intersection

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

    print(expr_sig.sum())
    print(allele_sig.sum())
    print(mpra_data['Label'].sum())
    mpra_data.drop(['Pool', 'pvalue_expr', 'padj_expr', 'pvalue_allele',
                    'padj_allele'], axis=1, inplace=True)
    mpra_data = mpra_data.loc[:, bg_cols]
    
    mpra_pos = mpra_data[mpra_data.Label == 1]
    neigh_pos = mpra_neighbors[mpra_data.Label == 1]
    mpra_original_neg = mpra_data[mpra_data.Label == 0]
    neigh_original_neg = mpra_neighbors[mpra_data.Label == 0]
    return mpra_pos, neigh_pos, mpra_original_neg, neigh_original_neg


def sample_mask(length, n):
    idx = np.random.choice(length, n, replace=False)
    mask = np.zeros(length)
    mask[idx] = 1
    return mask.astype(bool)


def setup(args):
    if args.background == 'unif':
        bg_proj = 'unif_background'
        proj_tag = 'unif_mixed'
    elif args.background == '1kg':
        bg_proj = '1kg_background'
        proj_tag = '1kg_mixed'

    # --- prepare mixed E116 --- #
    background_data, bg_neighbors = load_all_background(bg_proj, 'train')

    mpra_pos, neigh_pos, mpra_original_neg, neigh_original_neg = \
        prepare_all_e116_data()

    assert all(background_data.columns == mpra_pos.columns)

    # match negatives and remove from background pool
    mask = sample_mask(len(background_data), mpra_pos.shape[0] * 15)
    mpra_neg = background_data.loc[mask, :]
    mixed_data = pd.concat([mpra_pos, mpra_neg], axis=0)
    mixed_data.to_csv(
        PROCESSED_DIR / f'mpra_e116_{proj_tag}' / 'matrix_all.csv', index=False)

    # repeat for neighbors
    neigh_neg = bg_neighbors[mask, :]
    mixed_neigh = np.vstack([neigh_pos, neigh_neg])
    np.save(
        PROCESSED_DIR / f'mpra_e116_{proj_tag}' / 'neighbors_all', mixed_neigh)

    # remove selected background variants from the pool to avoid re-use
    background_data = background_data.loc[~mask, :]
    bg_neighbors = bg_neighbors[~mask, :]


    # --- prepare mixed nova --- #
    mpra_pos, neigh_pos, mpra_original_neg, neigh_original_neg = \
        prepare_all_nova_data(background_data.columns)

    # split into train and test sets
    mask = sample_mask(len(mpra_pos), int(0.75 * len(mpra_pos)))
    train_mpra_pos = mpra_pos.loc[mask, :]
    test_mpra_pos = mpra_pos.loc[~mask, :]
    train_neigh_pos = neigh_pos[mask, :]
    test_neigh_pos = neigh_pos[~mask, :]

    # match train negatives
    mask = sample_mask(len(background_data), train_mpra_pos.shape[0] * 15)
    mpra_neg = background_data.loc[mask, :]
    mixed_data = pd.concat([train_mpra_pos, mpra_neg], axis=0)
    mixed_data.to_csv(
        PROCESSED_DIR / f'mpra_nova_{proj_tag}' / 'matrix_train.csv', index=False)

    # repeat for neighbors
    neigh_neg = bg_neighbors[mask, :]
    mixed_neigh = np.vstack([train_neigh_pos, neigh_neg])
    np.save(
        PROCESSED_DIR / f'mpra_nova_{proj_tag}' / 'neighbors_train', mixed_neigh)

    # match test negatives
    background_data, bg_neighbors = load_all_background(bg_proj, 'test')
    mask = sample_mask(len(background_data), test_mpra_pos.shape[0] * 15)
    mpra_neg = background_data.loc[mask, :]
    mixed_data = pd.concat([test_mpra_pos, mpra_neg], axis=0)
    mixed_data.to_csv(
        PROCESSED_DIR / f'mpra_nova_{proj_tag}' / 'matrix_test.csv', index=False)

    # repeat for neighbors
    neigh_neg = bg_neighbors[mask, :]
    mixed_neigh = np.vstack([test_neigh_pos, neigh_neg])
    np.save(
        PROCESSED_DIR / f'mpra_nova_{proj_tag}' / 'neighbors_test', mixed_neigh)


    # # bundle orig negatives together here and save...
    # test_pos = test[test.Label == 1].copy()
    # print(test_pos.shape)
    # match_orig_neg = mpra_original_neg.sample(n=15*test_pos.shape[0], replace=False)
    # print(match_orig_neg.shape)
    # test_supp = pd.concat([test_pos, match_orig_neg], axis=0)
    # test_supp.to_csv(PROCESSED_DIR / 'mpra_nova_neg' / 'matrix_test.csv', index=False)


def setup_final_datasets(args):
    if args.background == 'unif':
        proj_tag = 'unif_mixed'
    elif args.background == '1kg':
        proj_tag = '1kg_mixed'
    
    final_dir = PROCESSED_DIR / f'final_{args.background}'

    e116_dir = PROCESSED_DIR / f'mpra_e116_{proj_tag}'
    e116 = pd.read_csv(e116_dir / 'matrix_all.csv')
    e116_neigh = np.load(e116_dir / 'neighbors_all.npy')

    nova_dir = PROCESSED_DIR / f'mpra_nova_{proj_tag}'
    nova_train = pd.read_csv(nova_dir / 'matrix_train.csv')
    nova_neigh_train = np.load(nova_dir / 'neighbors_train.npy')

    all_train = pd.concat([e116, nova_train], axis=0)
    all_train.to_csv(final_dir / 'matrix_train.csv', index=False)
    all_neigh_train = np.vstack([e116_neigh, nova_neigh_train])
    np.save(final_dir / 'neighbors_train', all_neigh_train)

    nova_test = pd.read_csv(nova_dir / 'matrix_test.csv')
    nova_test.to_csv(final_dir / 'matrix_test.csv', index=False)
    nova_neigh_test = np.load(nova_dir / 'neighbors_test.npy')
    np.save(final_dir / 'neighbors_test', nova_neigh_test)


def setup_genonet(args):
    from utils.data_utils import extract_genonet

    proj = f'final_{args.background}'
    test_data = pd.read_csv(PROCESSED_DIR / proj / 'matrix_test.csv')
    bed = test_data[['chr', 'pos']].copy()
    bed['pos_end'] = bed['pos'].astype(int) + 1
    bed['rs'] = [f'test_{x}' for x in range(bed.shape[0])]
    bed = bed[['chr', 'pos', 'pos_end', 'rs']]

    extract_genonet(bed, PROCESSED_DIR / proj / 'test_genonet.tsv', 'mean')


def validate_datasets(args):
    if args.background == 'unif':
        proj_tag = 'unif_mixed'
    elif args.background == '1kg':
        proj_tag = '1kg_mixed'

    e116_dir = PROCESSED_DIR / 'mpra_e116'
    nova_dir = PROCESSED_DIR / 'mpra_nova'
    final_dir = PROCESSED_DIR / f'final_{args.background}'

    ref = pd.read_csv(e116_dir / 'matrix_all.csv', nrows=2)
    ref2 = pd.read_csv(nova_dir / 'matrix_all.csv', nrows=2)
    train = pd.read_csv(final_dir / 'matrix_train.csv', nrows=2)
    test = pd.read_csv(final_dir / 'matrix_test.csv', nrows=2)

    assert all(ref.columns == ref2.columns)
    assert all(ref.columns == train.columns)
    assert all(ref.columns == test.columns)
    print('Validation completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', '-b', choices=['unif', '1kg'])
    args = parser.parse_args()

    setup(args)
    setup_final_datasets(args)
    validate_datasets(args)
    setup_genonet(args)
