''' Functions for extracting epigenetic features from bigwig files given input
bed file with genomic locations. '''

from subprocess import call
import glob
import os
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
from joblib import Parallel, delayed

import dask
from dask import delayed
import dask.dataframe as dd

from constants import *


def pull_roadmap_features(bedfile, feature_dir=TMP_DIR):
    """ For each ROADMAP marker, execute bigwig pulls for each tissue
    """
    # make tmp directory if not present
    os.makedirs(feature_dir, exist_ok=True)

    # clear tmp directory
    for f in os.listdir(feature_dir):
        try:
            os.remove(feature_dir / f)
        except IsADirectoryError:
            pass

    # save temporary bedfile with correct input format
    bedfile['chr'] = bedfile['chr'].map(lambda x: f'chr{x}')
    bedfile.to_csv(feature_dir / 'tmp.bed', sep='\t', header=False, index=False)

    # query each feature from bigwig
    for marker in ROADMAP_MARKERS:
        for i in range(1, 130):
            pull_command(marker, i, feature_dir / 'tmp.bed', feature_dir)
    return True

    # for marker in ROADMAP_MARKERS:
    #     Parallel(n_jobs=-1)(
    #         delayed(pull_command)(marker,
    #                               i,
    #                               feature_dir / 'tmp.bed',
    #                               feature_dir)
    #         for i in tqdm(range(1, 130))
    #     )


def compile_roadmap_features(bedfile, outpath, col_order,
                             feature_dir=TMP_DIR, keep_rs_col=False,
                             summarize='mean'):
    # stack features into dataframe
    compiled = features_to_csv(feature_dir, summarize=summarize)
    compiled = pd.merge(bedfile, compiled, left_on='rs', right_on='variant')

    # figure out column ordering
    col_order = ['chr', 'pos'] + col_order
    if keep_rs_col:
        compiled.drop(['variant', 'pos_end'], axis=1, inplace=True)
        col_order = col_order + ['rs']
    else:
        compiled.drop(['rs', 'variant', 'pos_end'], axis=1, inplace=True)

    compiled = compiled.loc[:, col_order]
    compiled.to_csv(outpath, sep='\t', index=False)


def pull_command(marker, i, bedfile, feature_dir=TMP_DIR):
    """ Extraction function to query a bedfile against a ROADMAP marker.

    Inputs:
        marker (str): ROADMAP marker (e.g. 'DNase')
        i (int): Tissue number, for example 58 will be formatted into E058
        bedfile (path): path to bedfile of variant locations to query

    Outputs are saved in a temporary storage directory to be aggregated later.
    """
    # for example: [...]/E058-DNase.imputed.pval.signal.bigwig
    filestr = ROADMAP_DIR / marker / f'E{i:03d}-{marker}{BIGWIG_TAIL}'
    output = feature_dir / f'{marker}-E{i:03d}'

    command = f'{BIGWIG_UTIL} -minMax {filestr} {bedfile} {output}'
    call(command, shell=True)


def features_to_csv(feature_dir=TMP_DIR, summarize='mean'):
    """ Post-extraction, combine temporary outputs into a DataFrame and save
    to csv.
    """
    features = [x for x in os.listdir(feature_dir) if re.search('E\d{3}$', x)]

    cols = []
    for fn in tqdm(features):
        tmp = pd.read_csv(feature_dir / f'{fn}',
            sep='\t',
            names=['variant', 'size', 'cov', 'sum', 'mean0', 'mean', 'min', 'max']
        )
        if summarize == 'mean':
            tmp = tmp.drop(['size', 'cov', 'sum', 'mean0', 'min', 'max'], axis=1)
            tmp = tmp.rename(columns={'mean': fn})
        elif summarize == 'max':
            tmp = tmp.drop(['size', 'cov', 'sum', 'mean0', 'mean', 'min'], axis=1)
            tmp = tmp.rename(columns={'max': fn})
        cols.append(tmp)

    df = pd.concat(cols, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def dask_features_to_csv(feature_dir=TMP_DIR):
    """ dask implementation can wait, there will be different implementation
    for neighbors"""
    def read_label_feature(fn):
        tmp = pd.read_csv(feature_dir / f'{fn}', sep='\t',
                            names=['variant', 'c1', 'c2', 'v1', 'v2', f'{fn}'])
        tmp = tmp.drop(['c1','c2','v1','v2'], axis=1)
        tmp['variant'] = tmp['variant'].astype(str)
        tmp[f'{fn}'] = tmp[f'{fn}'].astype(np.float32)
        tmp.set_index('variant', drop=True, inplace=True)
        return tmp

    df = dd.read_csv(os.path.join(feature_dir, '*-E*'), sep='\t',
                    names=['variant', 'c1', 'c2', 'v1', 'v2', 'data'],
                    include_path_column=True)
    df['path'] = df['path'].map(lambda x: Path(x).name)
    df = df.drop(['c1', 'c2', 'v1', 'v2'], axis=1)
    dd.pivot_table(df, index='variant', values='data', columns='path')

    # features = [x for x in os.listdir(feature_dir) if re.search('E\d{3}$', x)]
    # dfs = [delayed(read_label_feature)(fn) for fn in features]

    # dd.from_delayed