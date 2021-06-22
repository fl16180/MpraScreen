import os
import pandas as pd
import numpy as np
import io, subprocess
from tqdm import tqdm
# from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder

from utils.bigwig_utils import pull_roadmap_features, compile_roadmap_features
from constants import *


def load_mpra_data(dataset, benchmarks=False):
    ''' processes raw MPRA data files and optionally benchmark files '''
    mpra_files = MPRA_TABLE[dataset]
    data = pd.read_csv(MPRA_DIR / mpra_files[0], sep='\t')

    if dataset == 'mpra_nova':
        data.rename(columns={'chrom': 'chr'}, inplace=True)
        data['rs'] = data['chr'].map(str) + ':' + data['pos'].map(str)
        data.drop_duplicates(subset=['rs'], inplace=True)

    ### setup mpra/epigenetic data ###
    data_prepared = data.assign(chr=data['chr'].apply(lambda x: int(x[3:]))) \
                        .sort_values(['chr', 'pos']) \
                        .reset_index(drop=True)
    data_prepared[['chr', 'pos']] = data_prepared[['chr', 'pos']].astype(int)

    if benchmarks:
        if not mpra_files[1]:
            return data_prepared, None

        bench = pd.read_csv(MPRA_DIR / mpra_files[1], sep='\t')

        ### setup benchmark data ###
        # modify index column to extract chr and pos information
        chr_pos = (bench.reset_index()
                        .loc[:, 'index']
                        .str
                        .split('-', expand=True)
                        .astype(int))

        # update benchmark data with chr and pos columns
        bench_prepared = (bench.reset_index()
                            .assign(chr=chr_pos[0])
                            .assign(pos=chr_pos[1])
                            .drop('index', axis=1)
                            .sort_values(['chr', 'pos'])
                            .reset_index(drop=True))

        # put chr and pos columns in front for readability
        reordered_columns = ['chr', 'pos'] + bench_prepared.columns.values.tolist()[:-2]
        bench_prepared = bench_prepared[reordered_columns]
        return data_prepared, bench_prepared

    return data_prepared


def _read_bed(x, **kwargs):
    """ Helper function to parse output from a tabix query """
    return pd.read_csv(x, sep=r'\s+', header=None, index_col=False, **kwargs)


def extract_eigen(bedfile, outpath):
    """ Extract rows from Eigen file corresponding to variants in
    input bedfile. Command line piping adapted from
    https://github.com/mulinlab/regBase/blob/master/script/regBase_predict.py

    """
    if os.path.exists(outpath):
        os.remove(outpath)

    head_file = EIGEN_DIR / 'header_noncoding.txt'
    with open(head_file, 'r') as f:
        header = f.read()
    header = header.strip('\n').split('  ')

    command = f'tabix {EIGEN_DIR}/{EIGEN_BASE} '

    with open(outpath, 'w') as fp:
        fp.write('\t'.join(header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            args = f'{row.chr}:{row.pos}-{row.pos}'
            full_cmd = command.replace('XX', str(row.chr)) + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                features = _read_bed(io.StringIO(stdout.decode('utf-8')))
                features = features.astype(str).values.tolist()
                for fr in features:
                    fp.write('\t'.join(fr) + '\n')


def extract_regbase(bedfile, outpath):
    """ Extract rows from regBase file corresponding to variants in
    input bedfile.
    """
    if os.path.exists(outpath):
        os.remove(outpath)

    head_file = REGBASE_DIR / REGBASE
    header = pd.read_csv(head_file, sep=r'\s+', nrows=3).columns
    header = header.values.tolist()

    command = f'tabix {REGBASE_DIR}/{REGBASE} '

    with open(outpath, 'w') as fp:
        fp.write('\t'.join(header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            args = f'{row.chr}:{row.pos}-{row.pos}'
            full_cmd = command + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                features = _read_bed(io.StringIO(stdout.decode('utf-8')))
                features = features.astype(str).values.tolist()
                for fr in features:
                    fp.write('\t'.join(fr) + '\n')


def extract_roadmap(bedfile, outpath, project,
                    get_new=True, keep_rs_col=False, summarize='mean'):
    """ Extract Roadmap data. Currently special cases for each project.
    Some of the MPRA datasets already have Roadmap data. For new data,
    the Roadmap must be extracted and compiled separately.
    """
    if os.path.exists(outpath):
        os.remove(outpath)

    col_order = get_roadmap_col_order(order='tissue')

    if not get_new and project in STANDARD_MPRA:
        data = load_mpra_data(project)
        data.drop(['rs', 'Label'], axis=1, inplace=True)
        data.to_csv(outpath, sep='\t', index=False)

    else:
        loc = TMP_DIR / project
        success = pull_roadmap_features(bedfile, feature_dir=loc)
        if success:
            compile_roadmap_features(bedfile, outpath,
                                     col_order,
                                     feature_dir=loc,
                                     keep_rs_col=keep_rs_col,
                                     summarize=summarize)


def extract_genonet(bedfile, outpath, summarize='mean'):
    """ Extract rows from GenoNet file corresponding to variants in
    input bedfile.
    """
    if os.path.exists(outpath):
        os.remove(outpath)

    cols = ['GenoNet-E{:03d}'.format(x) for x in range(1, 130) if x not in [60, 64]]
    header = ['chr', 'pos_start', 'pos_end', 'reg_id'] + cols

    command = f'tabix {GENONET_DIR}/{GENONET_BASE} '

    with open(outpath, 'w') as fp:
        full_header = header + ['rs']
        fp.write('\t'.join(full_header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            args = f'chr{row.chr}:{row.pos}-{row.pos}'
            full_cmd = command.replace('XX', str(row.chr)) + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                features = _read_bed(io.StringIO(stdout.decode('utf-8')),
                                     na_values='.')
                features.columns = header
                features['reg_id'] = 0.0

                if summarize == 'mean':
                    features = features.groupby('chr', as_index=False).mean()
                elif summarize == 'max':
                    features = features.groupby('chr', as_index=False).max()

                features['pos_start'] = row.pos
                features['pos_end'] = row.pos_end
                features['rs'] = row.rs

                features = features.astype(str).values.tolist()
                for row in features:
                    fp.write('\t'.join(row) + '\n')


def clean_eigen_data(filename):
    eigen = pd.read_csv(filename, sep='\t', na_values='.')

    # manually convert scores to float due to NaN processing
    eigen.iloc[:, 4:] = eigen.iloc[:, 4:].astype(float)

    # average over variant substitutions
    eigen = eigen.rename(columns={'chr': 'chr', 'position': 'pos'}) \
                 .drop('alt', axis=1) \
                 .groupby(['chr', 'pos', 'ref'], as_index=False) \
                 .mean()
    eigen[['chr', 'pos']] = eigen[['chr', 'pos']].astype(int)
    return eigen


def clean_regbase_data(filename, keep_all=False):
    regbase = pd.read_csv(filename, sep='\t', na_values='.')

    # manually convert scores to float due to NaN processing
    regbase.iloc[:, 5:] = regbase.iloc[:, 5:].astype(float)
    
    if keep_all:
        regbase = regbase.rename(
            columns={'#Chrom': 'chr', 'Pos_end': 'pos', 'Ref': 'ref'})
        regbase['idx'] = regbase['chr'].astype(str) + '-' + regbase['pos'].astype(str)

        # one-hot encode ref
        regbase = pd.get_dummies(
            regbase, prefix=['is'], columns=['ref'], drop_first=False)

        pivot_cols = ['CADD', 'CADD_PHRED',
                      'DANN', 'DANN_PHRED',
                      'FATHMM-MKL', 'FATHMM-MKL_PHRED', 'FIRE',
                      'FIRE_PHRED', 'FATHMM-XF', 'FATHMM-XF_PHRED',
                      'CScape', 'CScape_PHRED', 'Orion']
        tmp = regbase[['idx', 'Alts'] + pivot_cols]

        tmp = tmp.drop_duplicates().pivot(index='idx', columns='Alts', values=pivot_cols)
        tmp.columns = ['-'.join(col).strip() for col in tmp.columns]
        tmp.reset_index(inplace=True)

        regbase = pd.merge(
            regbase.drop(pivot_cols, axis=1).drop_duplicates(subset='idx'),
            tmp, on='idx')
        regbase.drop(['Pos_start', 'Alts', 'idx'], axis=1, inplace=True)

    else:
        # average over variant substitutions
        regbase = regbase.rename(columns={'#Chrom': 'chr', 'Pos_end': 'pos', 'Ref': 'ref'}) \
                         .drop(['Pos_start', 'Alts'], axis=1) \
                         .groupby(['chr', 'pos', 'ref'], as_index=False) \
                         .mean()
    
    regbase[['chr', 'pos']] = regbase[['chr', 'pos']].astype(int)
    return regbase


def get_roadmap_col_order(order='tissue'):
    """
    Get list of roadmap features in specified order.

    Inputs:
        order: 'tissue' or 'marker'

    Converts column index to list for downstream appends """
    if order == 'tissue':
        data = load_mpra_data(ROADMAP_COL_ORDER_REF)
        data.drop(['chr', 'pos', 'rs', 'Label'], axis=1, inplace=True)
        return data.columns.tolist()
    elif order == 'marker':
        marks = ROADMAP_MARKERS
        nums = [x for x in range(1, 130) if x not in [60, 64]]
        cols = [x + '-E{:03d}'.format(y) for x in marks for y in nums]
        return cols


def get_tissue_scores(data, tissue='E116'):
    ids = ['chr', 'pos', 'rs', 'Label']
    feats = [x for x in data.columns if x not in ids and tissue in x]

    df_select = data[ids + feats]
    return df_select


def split_train_test(data, test_frac=0.2, seed=None):
    if seed:
        np.random.seed(seed)

    m  = data.shape[0]
    test_size = int(test_frac * m)
    perm = np.random.permutation(m)

    test = data.iloc[perm[:test_size], :]
    train = data.iloc[perm[test_size:], :]

    return train, test


def split_train_dev_test(data, dev_frac, test_frac, seed=None):
    if seed:
        np.random.seed(seed)

    m  = data.shape[0]
    dev_size = int(dev_frac * m)
    test_size = int(test_frac * m)
    perm = np.random.permutation(m)

    dev = data.iloc[perm[:dev_size], :]
    test = data.iloc[perm[dev_size:dev_size + test_size], :]
    train = data.iloc[perm[dev_size + test_size:], :]

    return train, dev, test


def rearrange_by_epigenetic_marker(df):
    marks = ROADMAP_MARKERS
    nums = [x for x in range(1, 130) if x not in [60, 64]]
    cols = [x + '-E{:03d}'.format(y) for x in marks for y in nums]
    return df.loc[:, cols]


def downsample_negatives(train, p):

    train_negs = train[train.Label == 0]
    train_pos = train[train.Label == 1]

    train_downsample = resample(train_negs,
                                replace=False,
                                n_samples=int(p * train_negs.shape[0]),
                                random_state=111)

    train_balanced = pd.concat([train_downsample, train_pos])
    return train_balanced.sample(frac=1, axis=0)


def upsample_positives(train, scale):

    train_negs = train[train.Label == 0]
    train_pos = train[train.Label == 1]

    train_upsample = resample(train_pos,
                              replace=True,
                              n_samples=scale * train_pos.shape[0],
                              random_state=111)

    train_balanced = pd.concat([train_negs, train_upsample])
    return train_balanced.sample(frac=1, axis=0)
