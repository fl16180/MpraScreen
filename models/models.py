import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from sklearn.isotonic import IsotonicRegression


class MpraDense(nn.Module):
    def __init__(self, n_input, n_units, nonlin=torch.sigmoid, dropout=0.3):
        super(MpraDense, self).__init__()
        self.nonlin = nonlin
        self.dense1 = nn.Linear(n_input, n_units[0])
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(n_units[0], n_units[1])
        self.final = nn.Linear(n_units[1], 2)

    def forward(self, X, return_embedding=False, **kwargs):
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense2(X))
        X = self.dropout(X)
        if return_embedding:
            return X
        X = F.softmax(self.final(X), dim=-1)
        return X


class MpraCNN(nn.Module):
    def __init__(self, n_filt, width=5, lin_units=100, input_dim=(127, 81)):
        super(MpraCNN, self).__init__()
        self.in_ch = input_dim[0]
        
        self.conv1 = nn.Conv1d(self.in_ch, n_filt, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(n_filt, n_filt, kernel_size=width)
        self.conv3 = nn.Conv1d(n_filt, n_filt, kernel_size=width)

        n_feat = self._get_conv_output(input_dim)
        self.dense1 = nn.Linear(n_feat, lin_units)
        self.dense2 = nn.Linear(lin_units, 2)

    def forward(self, X, **kwargs):
        X = self._forward_conv(X)
        X = X.view(X.size(0), -1)

        X = F.leaky_relu(self.dense1(X))
        X = F.softmax(self.dense2(X), dim=-1)
        return X

    def _forward_conv(self, X):
        X = X.view(X.shape[0], self.in_ch, 81)
        X = F.leaky_relu(F.max_pool1d(self.conv1(X), 2))
        X = F.leaky_relu(F.max_pool1d(self.conv2(X), 2))
        # X = F.leaky_relu(self.conv3(X))
        return X

    def _get_conv_output(self, shape):
        X_rand = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(X_rand)
        return output_feat.data.view(1, -1).size(1)


class MpraAutoEncoder(nn.Module):
    def __init__(self, nonlin=F.leaky_relu):
        self.encoder = nn.Sequential(
        )

        self.decoder = nn.Sequential(
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decode(encoded)
        return decoded, encoded
        
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded = y_pred
        return super().get_loss(decoded, y_true, *args, **kwargs)


class MpraFullCNN(nn.Module):
    def __init__(self,
                 width=5,
                 n_lin1=400,
                 n_lin2=256,
                 n_filt=16,
                 nonlin=F.leaky_relu,
                 conv_dim=(8, 81),
                 lin_dim=1071):
        super(MpraFullCNN, self).__init__()
        self.in_ch = conv_dim[0]
        self.nonlin = nonlin
        self.lin = lin_dim
        
        # input (N, 135, 81)
        self.conv1 = nn.Conv1d(self.in_ch, n_filt, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(n_filt, n_filt, kernel_size=width)
        self.conv3 = nn.Conv1d(n_filt, n_filt, kernel_size=width)

        self.dense_sc = nn.Linear(self.lin, n_lin1)
        # self.dense_sc = nn.Sequential(
        #     nn.Linear(self.lin, n_lin1),
        #     nn.Linear(n_lin1, n_lin1))


        n_feat = self._get_conv_output(conv_dim)

        self.dense1 = nn.Linear(n_feat + n_lin1, n_lin2)
        self.dense2 = nn.Linear(n_lin2, 2)

    def forward(self, X, return_embedding=False, **kwargs):
        X_neigh, X_score = self._reshape_conv(X)
        
        X_neigh = self._forward_conv(X_neigh)
        X_neigh = X_neigh.view(X_neigh.size(0), -1)

        X_score = torch.sigmoid(self.dense_sc(X_score))

        X = torch.cat([X_neigh, X_score], dim=1)
        X = torch.sigmoid(self.dense1(X))
        if return_embedding:
            return X
        X = F.softmax(self.dense2(X), dim=-1)
        return X

    def _reshape_conv(self, X):
        X_score = X[:, :self.lin]
        X_neigh = X[:, self.lin:].view(X.shape[0], 8, 81)
        return X_neigh, X_score

    def _forward_conv(self, X_neigh):
        # CNN layers on neighbor sequence
        X_neigh = self.nonlin(F.avg_pool1d(self.conv1(X_neigh), 2))
        X_neigh = self.nonlin(F.avg_pool1d(self.conv2(X_neigh), 2))
        return X_neigh

    def _get_conv_output(self, shape):
        X_rand = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(X_rand)
        return output_feat.data.view(1, -1).size(1)


class MpraMultiLabelCNN(nn.Module):
    def __init__(self,
                 width=5,
                 n_lin1=400,
                 n_lin2=256,
                 n_filt=16,
                 nonlin=F.leaky_relu,
                 conv_dim=(8, 81),
                 lin_dim=1071):
        super(MpraMultiLabelCNN, self).__init__()
        self.in_ch = conv_dim[0]
        self.nonlin = nonlin
        self.lin = lin_dim
        
        # input (N, 135, 81)
        self.conv1 = nn.Conv1d(self.in_ch, n_filt, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(n_filt, n_filt, kernel_size=width)
        self.conv3 = nn.Conv1d(n_filt, n_filt, kernel_size=width)

        self.dense_sc = nn.Linear(self.lin, n_lin1)

        n_feat = self._get_conv_output(conv_dim)

        self.dense1 = nn.Linear(n_feat + n_lin1, n_lin2)
        self.dense2 = nn.Linear(n_lin2, 2)

    def forward(self, X, **kwargs):
        X_neigh, X_score = self._reshape_conv(X)
        
        X_neigh = self._forward_conv(X_neigh)
        X_neigh = X_neigh.view(X_neigh.size(0), -1)

        X_score = torch.sigmoid(self.dense_sc(X_score))

        X = torch.cat([X_neigh, X_score], dim=1)
        X = torch.sigmoid(self.dense1(X))
        X = self.dense2(X)
        return X

    def _reshape_conv(self, X):
        X_score = X[:, :self.lin]
        X_neigh = X[:, self.lin:].view(X.shape[0], 8, 81)
        return X_neigh, X_score

    def _forward_conv(self, X_neigh):
        # CNN layers on neighbor sequence
        X_neigh = self.nonlin(F.avg_pool1d(self.conv1(X_neigh), 2))
        X_neigh = self.nonlin(F.avg_pool1d(self.conv2(X_neigh), 2))
        return X_neigh

    def _get_conv_output(self, shape):
        X_rand = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(X_rand)
        return output_feat.data.view(1, -1).size(1)


class Calibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, scores):
        sorted_scores = np.sort(scores)
        quants = (np.arange(len(scores)) + 1) / len(scores)
        self.model.fit(sorted_scores, quants)

    def transform(self, scores):
        return self.model.predict(scores)


# class MpraFullCNN(nn.Module):
#     def __init__(self,
#                  width=5,
#                  n_lin1=400,
#                  n_lin2=256,
#                  n_filt=16,
#                  nonlin=F.leaky_relu,
#                  conv_dim=(135, 81),
#                  lin_dim=945):
#         super(MpraFullCNN, self).__init__()
#         self.in_ch = conv_dim[0]
#         self.nonlin = nonlin
        
#         # input (N, 135, 81)
#         self.conv0 = nn.Conv1d(self.in_ch, n_filt, kernel_size=1)
#         self.conv1 = nn.Conv1d(n_filt, n_filt, kernel_size=4, padding=1)
#         self.conv2 = nn.Conv1d(n_filt, n_filt, kernel_size=width)
#         # self.conv3 = nn.Conv1d(16, 16, kernel_size=width)

#         self.dense_sc = nn.Linear(lin_dim, n_lin1)

#         n_feat = self._get_conv_output(conv_dim)
#         # print(n_feat)
#         self.dense1 = nn.Linear(n_feat + n_lin1, n_lin2)
#         self.dense2 = nn.Linear(n_lin2, 2)

#     def forward(self, X, **kwargs):
#         X_neigh, X_score = self._reshape_conv(X)
        
#         X_neigh = self._forward_conv(X_neigh)
#         X_neigh = X_neigh.view(X_neigh.size(0), -1)

#         X_score = torch.sigmoid(self.dense_sc(X_score))

#         X = torch.cat([X_neigh, X_score], dim=1)
#         X = torch.sigmoid(self.dense1(X))
#         X = F.softmax(self.dense2(X), dim=-1)
#         return X

#     def _reshape_conv(self, X):
#         X_score = X[:, :945]
#         X_neigh = X[:, 945:].view(X.shape[0], 135, 81)
#         return X_neigh, X_score

#     def _forward_conv(self, X_neigh):
#         # CNN layers on neighbor sequence
#         X_neigh = self.nonlin(self.conv0(X_neigh))
#         X_neigh = self.nonlin(F.avg_pool1d(self.conv1(X_neigh), 2))
#         X_neigh = self.nonlin(F.avg_pool1d(self.conv2(X_neigh), 2))
#         return X_neigh

#     def _get_conv_output(self, shape):
#         X_rand = Variable(torch.rand(1, *shape))
#         output_feat = self._forward_conv(X_rand)
#         return output_feat.data.view(1, -1).size(1)
