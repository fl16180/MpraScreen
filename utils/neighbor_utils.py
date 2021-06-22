import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
from tqdm import tqdm
from joblib import Parallel, delayed

# import dask
# import dask.dataframe as dd

from constants import *
from utils.bigwig_utils import pull_roadmap_features
from utils.data_utils import get_roadmap_col_order


def reshape_roadmap_files(feature_dir=TMP_DIR):
    """ Reshape Roadmap features extracted from bigwigs into rectangle
    form (variant x neighbors). Keeps as a separate file for each feature
    """ 
    if not os.path.exists(feature_dir / 'reshape'):
        os.mkdir(feature_dir / 'reshape')
    
    def process_feature_file(feature_dir, feat):
        tmp = pd.read_csv(
            feature_dir / feat,
            sep='\t',
            names=['variant', 'size', 'cov', 'sum', 'mean0', 'mean', 'min', 'max'])

        tmp.drop(['size', 'cov', 'sum', 'mean0', 'min', 'max'], axis=1, inplace=True)
        tmp = tmp.rename(columns={'mean': feat})

        tmp[['chr-pos', 'neighbor']] = tmp['variant'].str.split(';', n=1, expand=True)
        tmp['neighbor'] = tmp['neighbor'].astype(int)
        tmp = tmp.pivot(index='chr-pos', values=feat, columns='neighbor')
        tmp = tmp.loc[:, np.sort(tmp.columns)]
        tmp.reset_index().to_csv(feature_dir / 'reshape' / feat, index=False)

    features = [x for x in os.listdir(feature_dir) if re.search('E\d{3}$', x)]

    Parallel(n_jobs=-1)(delayed(process_feature_file)(feature_dir, f) for f in tqdm(features))


def add_bed_neighbors(bed, n_neighbor=40, sample_res=25):
    """ Takes a dataframe bedfile and adds entries for neighboring
    variants for each variant in the dataset.

    Inputs:
        bed (df): dataframe with 'chr', 'pos', and 'rs' entries
        n_neighbor (int): number of neighbors to add on each side of the variant
        sample_res (int): number of bp between each neighbor
    """
    nrange = n_neighbor * sample_res
    neighbors = np.arange(-nrange, nrange + 1, sample_res)
    rs_tag = np.array([f';{x}' for x in neighbors])

    chr_n = np.repeat(bed['chr'].values[:, None], len(neighbors), axis=1)
    pos_n = bed['pos'].values[:, None] + neighbors[None, :]
    pos_end_n = pos_n + 1
    rs_n = bed['rs'].values[:, None] + rs_tag[None, :]

    neigh_bed = pd.DataFrame({'chr': chr_n.flatten(),
                              'pos': pos_n.flatten(),
                              'pos_end': pos_end_n.flatten(),
                              'rs': rs_n.flatten()})
    return neigh_bed


def process_roadmap_neighbors(ref_dfs, splits, project_dir,
                              tissue='all', feature_dir=TMP_DIR):
    """ Compile reshaped bigwig pulls into numpy arrays for modeling.

    Reshaped roadmap features from bigwig should be in the
    'feature_dir/reshape' directory. Compiled files will be saved in
    project_dir in a subdirectory called neighbors.

    tissue='all' will generate separate train/test arrays for each
    roadmap marker with all tissues grouped together. Otherwise,
    only train/test arrays for the specific tissue will be made, with all
    roadmap markers grouped together for that tissue (e.g. tissue='E116')
    """
    assert len(ref_dfs) == len(splits)

    def process_batch(features, ref_dfs, ident):
        dfs = []
        for feat in features:
            df = pd.read_csv(feature_dir / 'reshape' / feat)
            dfs.append(df)

        for ref, spl in zip(ref_dfs, splits):
            arr_out = np.zeros((ref.shape[0], len(features), dfs[0].shape[1] - 1))

            for f, df in tqdm(enumerate(dfs)):
                tmp = pd.merge(ref['chr-pos'], df, on='chr-pos', how='left', sort=False)
                assert np.all(tmp['chr-pos'].values == ref['chr-pos'].values)
                tmp.drop('chr-pos', axis=1, inplace=True)

                arr_out[:, f, :] = tmp.values
            np.save(savepath / f'{spl}_{ident}', arr_out)

    savepath = Path(project_dir) / 'neighbors'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for ref in ref_dfs:
        ref['chr-pos'] = ref['chr'].map(str) + '-' + ref['pos'].map(str)

    # roadmap feature ordering (useful for organizing CNN)
    if tissue == 'all':
        features = {}
        for r in ROADMAP_MARKERS:
            features[r] = [r + '-E{:03d}'.format(x)
                           for x in range(1, 130)
                           if x not in [60, 64]]

        Parallel(n_jobs=1)(
            delayed(process_batch)(feats, ref_dfs, ident=r)
            for r, feats in features.items()
        )

    else:
        features = [f'{x}-{tissue}' for x in ROADMAP_MARKERS]
        process_batch(features, ref_dfs, ident=tissue)
            
    
def dask_stuff_backup():
    feature_dir = Path('./')
    df = dd.read_csv(os.path.join(feature_dir, '*-E*'), sep='\t',
                    names=['variant', 'c1', 'c2', 'v1', 'v2', 'data'],
                    include_path_column=True)
    df['path'] = df['path'].map(lambda x: Path(x).name)
    df = df.drop(['c1', 'c2', 'v1', 'v2'], axis=1)

    df = dd.pivot_table(df, index='variant', values='data', columns='path') 
    df['chr-pos'] = df['variant'].str.partition(';')[0]

    # df['chr'] = df['chrpos'].str.partition('-')[0]
    # df['pos'] = df['chrpos'].str.partition('-')[2]
    # df[['chr', 'pos']] = df[['chr', 'pos']].astype(int)
    # df = df.drop(['chrpos'], axis=1)

    ref = pd.DataFrame({'chr': [1, 1, 1], 'pos': [94986,706645,723891]})
    ref['chr-pos'] = ref['chr'].map(str) + '-' + ref['pos'].map(str)

    df_split = dd.merge(ref, df, on='chr-pos', how='left')

    features = ['DNase-E001','DNase-E120','H3K27ac-E100','DNase-E123']

        # dd.pivot_table(df, index='variant', values='data', columns='path') 

    for i, ref in enumerate(ref_dfs):
        ref = ref[['chr', 'pos']]
        df_split = dd.merge(ref, df, on=['chr', 'pos'], how='left')
        
        mat = split_to_mat(df_split, features)
    return


def split_to_mat(df_split, features):

    # current pos is neighbor position instead of variant position
    neighbors.drop(['chr', 'pos'], axis=1, inplace=True)

    # get variant chr/pos from reference bedfile
    neighbors[['rs', 'N']] = neighbors['rs'].str.split(';', n=1, expand=True)
    neighbors['N'] = neighbors['N'].astype(int)
    neighbors = pd.merge(ref_bed[['chr', 'pos', 'rs']], neighbors, on='rs')

    # pivot dataframe on neighbors to get wide form
    
    features = get_roadmap_col_order(order='marker')

    tmp = tmp.loc[:, features]

    # reorder index and columns to match ref_bed variants
    ordered_nbrs = np.sort(neighbors['N'].unique())
    col_order = [(feat, nbr) for feat in features for nbr in ordered_nbrs]
    tmp = tmp.reindex(ref_bed['rs'])

    n_neighbors = len(neighbors['N'].unique())
    feat_mat = tmp.values.reshape(tmp.shape[0], len(features), n_neighbors)

    # store feature matrix and chr/pos reference array
    np.save(outpath, feat_mat)
    np.save(outpath.with_suffix('.ref'), ref_bed[['chr', 'pos']].values)


if __name__ == '__main__':

    test_bed = {'chr': [1, 1, 3, 4],
                'pos': [12000, 13000, 4000, 1015],
                'pos_end': [12001, 13001, 4001, 1016],
                'rs': ['v1', 'v2', 'v3', 'v4']}

    bed = pd.DataFrame(test_bed)

    neigh_bed = add_bed_neighbors(bed, n_neighbor=2, sample_res=10)

    neigh_bed['feat1'] = np.arange(10, 10 + neigh_bed.shape[0])
    neigh_bed['feat2'] = np.arange(2.1, 2.1 + neigh_bed.shape[0])
    neigh_bed.drop(['pos_end'], axis=1, inplace=True)

    feature_dir = Path('./')

    df = dd.read_csv(os.path.join(feature_dir, '*-E*'), sep='\t',
                    names=['variant', 'c1', 'c2', 'v1', 'v2', 'data'],
                    include_path_column=True)
    df['path'] = df['path'].map(lambda x: Path(x).name)
    df = df.drop(['c1', 'c2', 'v1', 'v2'], axis=1)
