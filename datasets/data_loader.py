import pandas as pd
import numpy as np
from constants import PROCESSED_DIR, N_NEIGH, SAMPLE_RES, ROADMAP_MARKERS, EPS
from utils.data_utils import get_roadmap_col_order

from .processor import Processor


def load_data_set(project, split='all',
                  datasets=['roadmap', 'eigen', 'regbase'],
                  make_new=True):
    ''' Combines processed data sources to a compiled matrix.
    split: ('train', 'test', 'all') matching the saved label names
    '''
    proj_loc = PROCESSED_DIR / project

    if project in ['mpra_e116_mixed', 'mpra_nova_mixed']:
        make_new = False
    try:
        if make_new:
            raise Exception(f'Compiling {split} matrix.')
        dat = pd.read_csv(proj_loc / f'matrix_{split}.csv')

    except Exception as e:
        dat = pd.read_csv(proj_loc / f'{split}_label.csv')

        for ds in datasets:
            df = pd.read_csv(proj_loc / f'{split}_{ds}.csv')
            if 'ref' in df.columns:
                df.drop('ref', axis=1, inplace=True)
            dat = pd.merge(dat, df,
                           on=['chr', 'pos'], suffixes=('', '__y'))
            dat.drop(list(dat.filter(regex='__y$')), axis=1, inplace=True)

        dat.drop_duplicates(['chr', 'pos'], inplace=True)
        dat.to_csv(proj_loc / f'matrix_{split}.csv', index=False)
    return dat


def load_neighbors_set(project, split='all',
                       n_neigh=40, sample_res=25, tissue='E116'):
    proj_loc = PROCESSED_DIR / project / 'neighbors'
    fname = proj_loc / f'{split}_{n_neigh}_{sample_res}_{tissue}.npy'
    return np.load(fname)


def load_compiled_neighbors_set(project, split='all'):
    proj_loc = PROCESSED_DIR / project
    fname = proj_loc / f'neighbors_{split}.npy'
    return np.load(fname)


def load_full_project_data(project, split='test'):
    try:
        X_neighbor = load_compiled_neighbors_set(
            project, split=split)
    except FileNotFoundError:
        X_neighbor = load_neighbors_set(project,
                                        split=split,
                                        n_neigh=N_NEIGH,
                                        sample_res=SAMPLE_RES,
                                        tissue='E116')
    X = load_data_set(project, split=split, make_new=False)
    return X, X_neighbor


def load_and_preprocess(project, model, split='all', multi_label=False):

    print(f'Loading data for {model}:')

    if model in ['glm', 'standard']:
        df = load_data_set(project, split=split, make_new=False)
        roadmap_cols = get_roadmap_col_order(order='marker')
        
        df[roadmap_cols] = np.log(df[roadmap_cols] + EPS)

        proc = Processor(project)
        df = proc.fit_transform(df, na_thresh=0.05)
        proc.save(model)

        X = df.drop(['chr', 'pos', 'Label'], axis=1) \
              .values \
              .astype(np.float32)

    elif model == 'neighbors':
        try:
            X_neighbor = load_compiled_neighbors_set(
                project, split=split)
        except FileNotFoundError:
            X_neighbor = load_neighbors_set(project, 
                                            split=split,
                                            n_neigh=N_NEIGH,
                                            sample_res=SAMPLE_RES,
                                            tissue='E116')
        print(X_neighbor.shape)
        X_neighbor = np.log(X_neighbor.astype(np.float32) + EPS)

        df = load_data_set(project, split=split, make_new=False)
        print(df.iloc[:5, :5])
        roadmap_cols = get_roadmap_col_order(order='marker')
        df[roadmap_cols] = np.log(df[roadmap_cols] + EPS)
        print('check1: ', df.shape)

        proc = Processor(project)
        df = proc.fit_transform(df, na_thresh=0.05)
        proc.save(model)
        print('check2: ', df.shape)

        rm_cols = [f'{x}-E116' for x in ROADMAP_MARKERS]

        X_score = df.drop(['chr', 'pos', 'Label'] + rm_cols, axis=1) \
                    .values \
                    .astype(np.float32)

        X_neighbor = X_neighbor.reshape(
            X_neighbor.shape[0], X_neighbor.shape[1] * X_neighbor.shape[2])
        print('Data shape: ', X_score.shape, X_neighbor.shape)
        X = np.hstack((X_score, X_neighbor))

    elif model == 'e116_neigh':
        X_neighbor = load_neighbors_set(project, split=split,
                                        n_neigh=N_NEIGH,
                                        sample_res=SAMPLE_RES,
                                        tissue='e116')
        print(X_neighbor.shape)
        X = np.log(X_neighbor.astype(np.float32) + EPS)

        df = load_data_set(project, split='all',
                           make_new=False)

    if multi_label:
        expr_label = df['Label'].isin(['Both', 'Expr']).values.astype(float)
        alle_label = df['Label'].isin(['Both', 'Allele']).values.astype(float)
        y = np.stack([expr_label, alle_label], axis=1)
    else:
        y = df['Label'].values.astype(np.int64)

    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)
    return X, y


class BackgroundDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def get_batch(self, n):
        print('Background shape: ')
        print(self.X.shape)
        print(self.y.shape)

        idx = np.random.permutation(self.X.shape[0])
        select_idx, remain_idx = idx[:n], idx[n:]
        X_selected = self.X[select_idx, :]
        y_selected = self.y[select_idx]

        self.X = self.X[remain_idx, :]
        self.y = self.y[remain_idx]
        return X_selected, y_selected
