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


def fit_model(args, X, y):
    print(f'Fitting model for {args.model}:')
    print(X.shape, y.shape)

    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    apr = EpochScoring(scoring='average_precision', lower_is_better=False)
    lrs = LRScheduler(policy='StepLR', step_size=10, gamma=0.5)

    if args.model == 'glm':
        glm = LogitNet(alpha=0.5, n_lambda=50, n_jobs=-1)
        glm.fit(X, y)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
        net = LogitNet(alpha=0.5, n_lambda=1, lambda_path=[glm.lambda_best_])

    else:
        net = model_loader[args.model]

    np.random.seed(1000)
    torch.manual_seed(1000)
    net.fit(X, y)
    save_model(net, args.project, args.model)

    scores = net.predict_proba(X)
    AUC = roc_auc_score(y, scores[:, 1])
    APR = average_precision_score(y, scores[:, 1])
    print('\tAUC ', np.round(AUC, 4))
    print('\tAPR ', np.round(APR, 4))


def evaluate_model(args, X_test, y_test):
    print(f"Evaluating model for {args.model}:")
    net = load_model(args.project, args.model)
    score_pred = net.predict_proba(X_test)

    print('\tAUROC: ', roc_auc_score(y_test, score_pred[:, 1]))
    print('\tAUPR: ', average_precision_score(y_test, score_pred[:, 1]))


def save_embeddings(args, X_test, y_test):
    print(f"Obtaining embeddings for {args.model}:")
    net = load_model(args.project, args.model)
    output = net.infer(X_test, return_embedding=True)
    output = output.detach().numpy()
    np.save(
        PROCESSED_DIR / args.project / 'output' / f'embeddings_{args.model}.npy', output
    )


def fit_calibrator(args):
    print(f"Fitting calibrator for {args.model}:")
    me = MultipleEval(['1kg_background2', '1kg_background3', '1kg_background4',
                       '1kg_background5', '1kg_background6'],
                       trained_data=args.project)
    me.eval_all()
    me.calibrate(score=f'NN_{args.model}')
    me.save_calibrator(score=args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', choices=PROJ_CHOICES,
                        default='final_1kg',
                        help='Project location for final model')
    parser.add_argument('--model', '-m', default='standard',
                        choices=['standard', 'neighbors'],
                        help='Model to train on')
    parser.add_argument('--evaluate', '-e', action='store_true', default=False,
                        help='Evaluate model on test set after fitting')
    args = parser.parse_args()

    X_train, y_train = load_and_preprocess(args.project, args.model, split='train')

    fit_model(args, X_train, y_train)

    if args.evaluate:
        evlt = Evaluator(trained_data=args.project, eval_data=args.project)
        evlt.setup_data(args.model, split='test')
        X_test, y_test = evlt.X, evlt.y

        # test set
        print('Test set: ')
        evaluate_model(args, X_test, y_test)

        # training set groups
        print('Train set: ')
        evaluate_model(args, X_train, y_train)

        # save embeddings
        save_embeddings(args, X_test, y_test)
        
        # fit calibrator over background pool
        if args.model == 'neighbors':
            fit_calibrator(args)
