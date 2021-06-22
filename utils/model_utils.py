import os
import pickle
import numpy as np
from scipy.stats import cauchy
from constants import PROCESSED_DIR

from constants import *
from datasets import *
from utils.data_utils import get_roadmap_col_order
from models import Calibrator


def save_model(model, project, name):
    fname = PROCESSED_DIR / project / 'models' / f'saved_model_{name}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(model, f)


def load_model(project, name):
    fname = PROCESSED_DIR / project / 'models' / f'saved_model_{name}.pkl'
    with open(fname, 'rb') as f:
        model = pickle.load(f)
    return model


def load_calibrator(project, name):
    fname = PROCESSED_DIR / project / 'models' / f'calibrator_{name}.pkl'
    with open(fname, 'rb') as f:
        model = pickle.load(f)
    return model


def get_cauchy_p(p_mat):
    ''' Vectorized version of the Cauchy combination test.
    
    p_mat is an (n x d) array, where n are tested variants and d are
    different p-values ''' 
    p_mat[p_mat > 0.99] = 0.99
    is_small = p_mat < 1e-16
    is_regular = p_mat >= 1e-16

    tmp = np.zeros_like(p_mat)
    tmp[is_small] = np.pi / p_mat[is_small]
    tmp[is_regular] = np.tan((0.5 - p_mat[is_regular]) * np.pi)

    cct_stat = np.nanmean(tmp, axis=1)
    large_val = cct_stat > 1e15
    cct_stat[large_val] = np.pi / cct_stat[large_val]
    cct_stat[~large_val] = 1 - cauchy.cdf(cct_stat[~large_val])
    return cct_stat


def concat_addl_scores(project, split='all', na_thresh=0.05):
    """ output score file contains chr, pos, label, nn_scores. For evaluation
    we want additional scores for regbase, eigen, etc. Concat these to the
    score file, excluding non-E116 roadmap scores.
    """
    proj_dir = PROCESSED_DIR / project

    scores = pd.read_csv(proj_dir / 'output' / f'nn_preds_{project}.csv',
                         sep=',')    
    addl = pd.read_csv(proj_dir / f'matrix_{split}.csv', sep=',')
    assert all(scores.pos == addl.pos)

    addl.drop(['chr', 'pos', 'Label'], axis=1, inplace=True)
    omit_roadmap = [x for x in get_roadmap_col_order() if x[-3:] != '116']
    addl.drop(omit_roadmap, axis=1, inplace=True)

    # drop scores that have >5% NaNs from metrics (were dropped from nn as well)
    na_filt = (addl.isna().sum() > na_thresh * len(addl))
    omit_cols = addl.columns[na_filt].tolist()
    omit_cols += [x + '_PHRED' for x in omit_cols if x + '_PHRED' in addl.columns]
    addl.drop(omit_cols, axis=1, inplace=True)

    scores = pd.concat([scores, addl], axis=1)
    return scores


class Evaluator:
    def __init__(self, trained_data='mpra_e116',
                 eval_data='mpra_nova'):
        self.trained_data = trained_data
        self.eval_data = eval_data
        self.ref = None
        self.model = None
        self.split = None
        self.X = None
        self.y = None
        self.scores = None
    
    def setup_data(self, model='standard', split='all', multi_label=False):
        self.model = model
        self.split = split

        if model in ['glm', 'standard']:
            df = load_data_set(self.eval_data, split=split, make_new=False)
            roadmap_cols = get_roadmap_col_order(order='marker')

            df[roadmap_cols] = np.log(df[roadmap_cols] + EPS)

            proc = Processor(self.trained_data)
            proc.load(model)
            df = proc.transform(df)

            self.ref = df[['chr', 'pos', 'Label']].copy()
            X = df.drop(['chr', 'pos', 'Label'], axis=1) \
                .values \
                .astype(np.float32)

        elif model == 'neighbors':
            try:
                X_neighbor = load_compiled_neighbors_set(
                    self.eval_data, split=split)
            except FileNotFoundError:
                X_neighbor = load_neighbors_set(self.eval_data,
                                                split=split,
                                                n_neigh=N_NEIGH,
                                                sample_res=SAMPLE_RES,
                                                tissue='E116')

            X_neighbor = np.log(X_neighbor.astype(np.float32) + EPS)

            df = load_data_set(self.eval_data, split=split, make_new=False)
            roadmap_cols = get_roadmap_col_order(order='marker')
            df[roadmap_cols] = np.log(df[roadmap_cols] + EPS)

            proc = Processor(self.trained_data)
            proc.load(model)
            df = proc.transform(df)

            rm_cols = [f'{x}-E116' for x in ROADMAP_MARKERS]

            X_score = df.drop(['chr', 'pos', 'Label'] + rm_cols, axis=1) \
                        .values \
                        .astype(np.float32)

            X_neighbor = X_neighbor.reshape(
                X_neighbor.shape[0], X_neighbor.shape[1] * X_neighbor.shape[2])
            X = np.hstack((X_score, X_neighbor))

        if multi_label:
            expr_label = df['Label'].isin(['Both', 'Expr']).values.astype(float)
            alle_label = df['Label'].isin(['Both', 'Allele']).values.astype(float)
            y = np.stack([expr_label, alle_label], axis=1)
        else:
            y = df['Label'].values.astype(np.int64)

        self.X = X
        self.y = y
        print('X.shape: ', self.X.shape)
        print('y.shape: ', self.y.shape)

    def predict_model(self):
        net = load_model(self.trained_data, self.model)
        print(self.X.shape)
        scores = net.predict_proba(self.X)
        self.scores = scores[:, 1]
        # expr_score = scores[:, 1, 0]
        # alle_score = scores[:, 1, 1]
        # self.scores = (expr_score, alle_score)

    def save_scores(self):
        proj_dir = PROCESSED_DIR / self.eval_data
        if not os.path.exists(proj_dir / 'output'):
            os.makedirs(proj_dir / 'output')

        try:
            df = pd.read_csv(proj_dir / 'output' / f'nn_preds_{self.eval_data}.csv',
                            sep=',')
            # assert np.all(self.y == df.Label)
        except (FileNotFoundError, ValueError) as e:
            print(e)
            df = pd.read_csv(proj_dir / f'matrix_{self.split}.csv', sep=',')
            cols = ['chr', 'pos', 'Label']
            df = df.loc[:, cols]
        
        print(df.shape, self.scores.shape)
        df[f'mpra_screen'] = self.scores
        # df['expr_screen'] = self.scores[0]
        # df['allele_screen'] = self.scores[1]

        df.to_csv(proj_dir / 'output' / f'nn_preds_{self.eval_data}.csv',
                sep=',', index=False)


class MultipleEval:
    def __init__(self, files, trained_data='gnom_mpra_mixed', model='neighbors'):
        self.files = files
        self.trained_data = trained_data
        self.cur = 0
        self.model = model

    def eval_single_file(self):
        evl = Evaluator(
            trained_data=self.trained_data, eval_data=self.files[self.cur])
        evl.setup_data(model=self.model, split='all')
        evl.predict_model()
        evl.save_scores()

        scores = concat_addl_scores(self.files[self.cur], split='all')
        scores.index.names = ['scores']
        scores.to_csv(
            PROCESSED_DIR / self.files[self.cur] / f'output/all_scores_nn_preds_{self.files[self.cur]}.csv',
            index=False)

    def eval_all(self):
        while self.cur < len(self.files):
            self.eval_single_file()
            self.cur += 1

    def _load_score(self, score):
        self.mem = np.array(())

        for f in self.files:
            tmp = pd.read_csv(PROCESSED_DIR / f / f'output/all_scores_nn_preds_{f}.csv')
            self.mem = np.append(self.mem, tmp[score].values)

    def calibrate(self, score='NN_neighbors'):
        self.score = score
        self._load_score(score)
        self.calib = Calibrator()
        self.calib.fit(self.mem)

    def load_holdout_prediction(self, holdout):
        df = pd.read_csv(
            PROCESSED_DIR / holdout / f'output/all_scores_nn_preds_{holdout}.csv'
        )
        self.probs = self.calib.transform(df[self.score].values)
        self.label = df['Label'].values

    def estimate_recall(self, thresholds):
        recalls = []
        for thr in thresholds:
            pred = (self.probs > thr).astype(int)
            rec = np.sum(self.label + pred == 2) / np.sum(self.label == 1)
            recalls.append(rec)
        return np.array(recalls)

    def value_to_percentile(self, val, score='NN_neighbors'):
        self._load_score(score)
        calib = Calibrator()
        calib.fit(self.mem)
        return calib.transform([val])

    def percentile_to_value(self, p, score='NN_neighbors'):
        self._load_score(score)
        sorted_score = np.sort(self.mem)
        tmp = sorted_score[int(p * len(sorted_score))]
        print(np.mean(self.mem <= tmp))
        return tmp

    def save_calibrator(self, score='neighbors'):
        fname = PROCESSED_DIR / self.trained_data / 'models' / f'calibrator_{score}.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self.calib, f)
