import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from constants import *
from datasets import load_and_preprocess
from utils.model_utils import save_model
from utils.data_utils import get_roadmap_col_order
import models

import torch
import torch.nn.functional as F
import optuna
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler

MODEL_CHOICES = ['glm', 'standard', 'neighbors', 'e116_neigh']


def main(args):
    X, y = load_and_preprocess(args.project, args.model, split='train')

    if args.model == 'e116_neigh':
        def objective(trial):
            auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
            apr = EpochScoring(scoring='average_precision', lower_is_better=False)
            lrs = LRScheduler(policy='StepLR', step_size=10, gamma=0.5)
            
            bs = trial.suggest_categorical('batch_size', [128])
            l2 = trial.suggest_uniform('l2', 5e-5, 1e-2)
            lr = trial.suggest_uniform('lr', 1e-4, 5e-3)
            epochs = trial.suggest_categorical('epochs', [30])
            n_filt = trial.suggest_categorical('n_filt', [8, 16, 32])
            width = trial.suggest_categorical('width', [3, 5, 7])
            lin_units = trial.suggest_categorical('lin_units', [100, 200, 400])

            net = NeuralNetClassifier(
                models.MpraCNN,

                optimizer=torch.optim.Adam,
                optimizer__weight_decay=l2,
                lr=lr,
                batch_size=bs,
                max_epochs=epochs,

                module__n_filt=n_filt,
                module__width=width,
                module__lin_units=lin_units,

                callbacks=[auc, apr],
                iterator_train__shuffle=True,
                train_split=None,
                verbose=0
            )
        
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)
            np.random.seed(1000)
            torch.manual_seed(1000)
            cv_scores = cross_val_predict(net, X, y, cv=kf,
                                        method='predict_proba', n_jobs=-1)
            return roc_auc_score(y, cv_scores[:, 1])
    
    elif args.model == 'neighbors':
        def objective(trial):
            auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
            apr = EpochScoring(scoring='average_precision', lower_is_better=False)
            lrs = LRScheduler(policy='StepLR', step_size=10, gamma=0.5)
            
            bs = trial.suggest_categorical('batch_size', [256])
            l2 = trial.suggest_uniform('l2', 5e-5, 5e-4)
            lr = trial.suggest_uniform('lr', 5e-5, 5e-4)
            epochs = trial.suggest_categorical('epochs', [30, 40])
            n_filt = trial.suggest_categorical('n_filt', [8, 16, 32])
            width = trial.suggest_categorical('width', [5])
            n_lin1 = trial.suggest_categorical('n_lin1', [400, 600])
            n_lin2 = trial.suggest_categorical('n_lin2', [400])

            net = NeuralNetClassifier(
                models.MpraFullCNN,

                optimizer=torch.optim.Adam,
                optimizer__weight_decay=l2,
                lr=lr,
                batch_size=bs,
                max_epochs=epochs,

                module__n_filt=n_filt,
                module__width=width,
                module__n_lin1=n_lin1,
                module__n_lin2=n_lin2,
                module__nonlin=F.leaky_relu,

                callbacks=[auc, apr],
                iterator_train__shuffle=True,
                train_split=None,
                verbose=0
            )
        
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)
            np.random.seed(1000)
            torch.manual_seed(1000)
            cv_scores = cross_val_predict(net, X, y, cv=kf,
                                        method='predict_proba', n_jobs=-1)
            return roc_auc_score(y, cv_scores[:, 1])
    print('Starting trials')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', choices=PROJ_CHOICES, default='mpra_e116')
    parser.add_argument('--model', '-m', default='standard', choices=MODEL_CHOICES,
                        help='Which data/model to train on')
    parser.add_argument('--iter', '-i', type=int,
                        help='Number of search iterations')
    args = parser.parse_args()

    main(args)
