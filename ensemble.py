import argparse
import numpy as np
import pandas as pd

# hide sklearn deprecation message triggered within skorch
from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)

import torch
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from glmnet import LogitNet

from constants import *
from datasets import load_and_preprocess
from models import model_loader
from utils.model_utils import save_model, load_model, Evaluator, MultipleEval
from utils.data_utils import get_roadmap_col_order

MODEL_CHOICES = ['glm', 'standard', 'neighbors']


def evaluate_model(args, X_test, y_test):
    print(f"Evaluating model for {args.model}:")
    net = load_model(args.project, args.model)
    score_pred = net.predict_proba(X_test)

    print('\tAUROC: ', roc_auc_score(y_test, score_pred[:, 1]))
    print('\tAUPR: ', average_precision_score(y_test, score_pred[:, 1]))


if __name__ == '__main__':

    evlt = Evaluator(trained_data='gnom_mpra_mixed', eval_data='gnom_mpra_mixed')
    evlt.setup_data('neighbors', split='test')
    X_test, y_test = evlt.X, evlt.y

    scores = np.zeros((len(y_test), 5))
    for i in range(1, 6):

        net = load_model(args.project, f'neighbors_{i}')
        score_pred = net.predict_proba(X_test)[:, 1]
        scores[:, i] = score_pred

        print(f'model {i}:')
        print('\tAUROC: ', roc_auc_score(y_test, score_pred))
        print('\tAUPR: ', average_precision_score(y_test, score_pred))

    ens_scores = np.mean(scores, 0)
    print(f'Ensemble:')
    print('\tAUROC: ', roc_auc_score(y_test, ens_scores))
    print('\tAUPR: ', average_precision_score(y_test, ens_scores))
