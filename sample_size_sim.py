import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle

# hide sklearn deprecation message triggered within skorch
from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from constants import *
from datasets import load_and_preprocess, BackgroundDataset
import models
from models.model_params import model_loader
from utils.model_utils import Evaluator

MODEL_CHOICES = ['standard', 'neighbors']


class Simulator:
    def __init__(self, model='standard', n_iter=12, strat='submit',
                 n_submit=100, n_match=15, n_select=2000, background=None):
        self.model = model
        self.n_iter = n_iter
        self.strat = strat
        self.n_submit = n_submit
        self.n_match = n_match
        self.n_select = n_select
        self.background = background

    def run_simulation(self, X_init, y_init, X_eval, y_eval, X_holdout, y_holdout):
        self.X_train = X_init
        self.y_train = y_init
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.X_holdout = X_holdout
        self.y_holdout = y_holdout

        self.progress = {}
        for i in range(self.n_iter):
            print(f'\nIter {i + 1}: ')
            self.progress[i] = {'n_train': len(self.y_train)}
            self.fit_model()
            self.score_holdout(i)
            selected = self.prioritize_variants(i)
            self.rebalance_new_datasets(selected)
            print(self.progress[i])
            if self.progress[i]['n_remaining'] == 0:
                break
        return self.progress

    def fit_model(self):
        self.net = model_loader[self.model]
        torch.manual_seed(1000)
        self.net.fit(self.X_train, self.y_train)

    def score_holdout(self, itr):
        test_scores = self.net.predict_proba(self.X_holdout)[:, 1]
        AUC = roc_auc_score(self.y_holdout, test_scores)
        APR = average_precision_score(self.y_holdout, test_scores)
        print('\tHoldout AUC ', np.round(AUC, 4))
        print('\tHoldout APR ', np.round(APR, 4))
        self.progress[itr]['test_AUC'] = np.round(AUC, 4)
        self.progress[itr]['test_APR'] = np.round(APR, 4)

    def prioritize_variants(self, itr):
        val_scores = self.net.predict_proba(self.X_eval)[:, 1]
        # AUC = roc_auc_score(self.y_eval, val_scores)
        # APR = average_precision_score(self.y_eval, val_scores)

        if len(val_scores) <= self.n_select and args.strat != 'submit':
            # if fewer remaining than validation size then select all
            selected = np.ones(len(val_scores)).astype(bool)
        elif args.strat == 'submit':
            selected = np.zeros_like(val_scores, dtype=bool)
            mask = np.where(self.y_eval == 1)[0]
            idx = np.random.choice(
                mask, min(self.n_submit, len(mask)), replace=False)
            selected[idx] = True
        elif self.strat in ['score', 'match']:
            # select high-score variants
            thresh_score = sorted(val_scores, reverse=True)[self.n_select]
            selected = val_scores > thresh_score

        # self.progress[itr]['remain_AUC'] = np.round(AUC, 4)
        # self.progress[itr]['remain_APR'] = np.round(APR, 4)
        self.progress[itr]['n_discovered'] = np.sum(self.y_eval[selected])
        self.progress[itr]['n_remaining'] = np.sum(self.y_eval[~selected])
        return selected

    def rebalance_new_datasets(self, selected):
        if self.strat in ['match', 'submit']:
            selected_pos = selected & (self.y_eval == 1)
            X_new_pos = self.X_eval[selected_pos, :]
            y_new_pos = self.y_eval[selected_pos]

            n_pos = np.sum(selected_pos)
            X_new_neg, y_new_neg = self.background.get_batch(
                self.n_match * n_pos)

            X_new = np.vstack([X_new_pos, X_new_neg])
            y_new = np.concatenate([y_new_pos, y_new_neg])
        elif self.strat == 'score':
            X_new = self.X_eval[selected, :]
            y_new = self.y_eval[selected]

        self.X_train = np.vstack([self.X_train, X_new])
        self.y_train = np.concatenate([self.y_train, y_new])
        self.X_eval = self.X_eval[~selected, :]
        self.y_eval = self.y_eval[~selected]


if __name__ == '__main__':
    np.random.seed(1111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='standard', choices=MODEL_CHOICES,
                        help='Which data/model to train on')
    parser.add_argument('--background', '-b', default='1kg',
                        choices=['unif', '1kg'],
                        help='Background pool')
    parser.add_argument('--max_iter', '-n', default=30, type=int,
                        help='Max learning iterations')
    parser.add_argument('--strat', '-s', choices=['score', 'random', 'background', 'match', 'submit'],
                        default='submit',
                        help='Strategy to prioritize variants')
    parser.add_argument('--n_submit', default=100,
                        help='Number of positives to add')
    parser.add_argument('--n_match', default=15,
                        help='Number of background controls per positive')
    parser.add_argument('--eval', default='nova-mixed',
                        choices=['nova-mixed', 'nova-only'])
    args = parser.parse_args()

    # --- Initial setup (setup mpra_nova eval and background data) --- #
    print('loading datasets: ')

    # load all variants
    evlt = Evaluator(trained_data=f'gnom_mpra_mixed',
        eval_data=f'gnom_mpra_mixed'
    )
    evlt.setup_data(args.model, split='train')
    X_train, y_train = evlt.X, evlt.y

    # get background matching set
    neg_mask = (y_train == 0)
    X_background = X_train[neg_mask]
    y_background = y_train[neg_mask]
    bgData = BackgroundDataset(X_background, y_background)

    # get positives set and split to init and pool
    pos_mask = (y_train == 1)
    X_pos = X_train[pos_mask]
    y_pos = y_train[pos_mask]
    X_init, X_pool, y_init, y_pool = train_test_split(
        X_pos, y_pos, test_size=0.8715)         # 200 pos, 3000 total

    # match init with backgrounds
    X_bg_init, y_bg_init = bgData.get_batch(len(y_init) * 14)
    X_init = np.vstack([X_init, X_bg_init])
    y_init = np.concatenate([y_init, y_bg_init])

    # prepare holdout validation set
    print('setting up holdout set: ')
    evlt = Evaluator(trained_data=f'gnom_mpra_mixed',
        eval_data=f'gnom_mpra_mixed'
    )
    evlt.setup_data(args.model, split='test')
    X_holdout, y_holdout = evlt.X, evlt.y

    # from pdb import set_trace; set_trace()

    print('\nStarting: ', np.sum(y_init), len(y_init))
    print('Total candidates: ', len(y_pool))
    print('Total significant to be discovered: ', np.sum(y_pool))

    print('\n--- starting simulation ---')
    sim = Simulator(model=args.model,
                    n_iter=args.max_iter,
                    strat=args.strat,
                    n_submit=args.n_submit,
                    n_match=args.n_match,
                    background=bgData)
    nova_summary = sim.run_simulation(
        X_init, y_init,
        X_pool, y_pool,
        X_holdout, y_holdout)

    nova_table = pd.DataFrame.from_dict(nova_summary, orient='index')
    nova_table['iter'] = np.arange(nova_table.shape[0])

    table = nova_table
    print(table)
    table.to_csv('./analysis/sample_size_sim.csv', index=False)
