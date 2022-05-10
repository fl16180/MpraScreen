import torch
from torch import nn
from skorch import NeuralNet, NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EpochScoring, LRScheduler

import models

auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
apr = EpochScoring(scoring='average_precision', lower_is_better=False)
lrs = LRScheduler(policy='StepLR', step_size=10, gamma=0.5)


MpraDense = NeuralNetClassifier(
    models.MpraDense,
    
    batch_size=256,
    optimizer=torch.optim.Adam,
    optimizer__weight_decay=2e-6,
    lr=1e-4,
    max_epochs=20,
    module__n_input=1079,
    module__n_units=(400, 250),
    module__dropout=0.3,
    
    callbacks=[auc, apr],
    iterator_train__shuffle=True,
    train_split=None
)


MpraFullCNN = NeuralNetClassifier(
    models.MpraFullCNN,

    optimizer=torch.optim.Adam,
    optimizer__weight_decay=5e-04,
    batch_size=128,
    lr=1e-4,
    max_epochs=30,

    module__n_filt=32,
    module__width=5,
    module__n_lin1=400,
    module__n_lin2=400,
    
    callbacks=[auc, apr],
    iterator_train__shuffle=True,
    train_split=None
)

# for pos-neg
MpraFullCNN = NeuralNetClassifier(
    models.MpraFullCNN,

    optimizer=torch.optim.Adam,
    optimizer__weight_decay=5e-04,
    batch_size=128,
    lr=2e-4,
    max_epochs=40,

    module__n_filt=32,
    module__width=5,
    module__n_lin1=400,
    module__n_lin2=400,
    
    callbacks=[auc, apr],
    iterator_train__shuffle=True,
    train_split=None
)


class MultiLabelNet(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        alpha = 1e-5
        pos_wt = 5.0

        y_true = y_true.float()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_wt, pos_wt]))

        l1_loss = alpha * torch.norm(self.module_.dense_sc.weight, p=1)
        return criterion(y_pred, y_true) + l1_loss


MpraMultiLabelCNN = NeuralNet(
    models.MpraMultiLabelCNN,

    criterion=nn.BCEWithLogitsLoss,

    optimizer=torch.optim.Adam,
    optimizer__weight_decay=5e-5,
    batch_size=256,
    lr=5e-4,
    max_epochs=30,

    module__n_filt=32,
    module__width=5,
    module__n_lin1=400,
    module__n_lin2=400,

    callbacks=[auc, apr],
    iterator_train__shuffle=True,
    train_split=None
)


model_loader = {
    'standard': MpraDense,
    'neighbors': MpraFullCNN,
    # 'neighbors': MpraMultiLabelCNN
}
