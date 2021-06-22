import os
import pickle as pkl
from constants import PROCESSED_DIR
from sklearn.preprocessing import StandardScaler


class Processor:
    def __init__(self, project):
        self.project = project
        self.omit_cols = []
        self.col_means = []
        self.fit_feats = []
        self.scaler = StandardScaler()

    def fit_transform(self,
                      df,
                      non_feats=['chr', 'pos', 'Label'],
                      na_thresh=0.05):

        # NA threshold and mean impute
        na_filt = (df.isna().sum() > na_thresh * len(df))
        omit_cols = df.columns[na_filt].tolist()
        omit_cols += [x + '_PHRED' for x in omit_cols if x + '_PHRED' in df.columns]

        # # drop scores that have a standardized PHRED version
        # non_phred_scores = ['Eigen-raw', 'Eigen-PC-raw', 'CADD', 'DANN',
        #                     'FATHMM-MKL', 'FunSeq2', 'Eigen', 'Eigen_PC',
        #                     'GenoCanyon', 'FIRE', 'ReMM', 'LINSIGHT',
        #                     'fitCons', 'FATHMM-XF', 'CScape', 'DVAR',
        #                     'FitCons2', 'ncER', 'PAFA']
        # omit_cols += non_phred_scores

        df.drop(omit_cols, axis=1, inplace=True)

        col_means = df.mean()
        df.fillna(col_means, inplace=True)

        # standardize feature columns
        fit_feats = list(set(df.columns) - set(non_feats))
        df[fit_feats] = self.scaler.fit_transform(df[fit_feats])

        self.omit_cols = omit_cols
        self.col_means = col_means
        self.fit_feats = fit_feats
        return df

    def transform(self, df):
        df.drop(df.columns.intersection(self.omit_cols), axis=1, inplace=True)
        df.fillna(self.col_means, inplace=True)
        df[self.fit_feats] = self.scaler.transform(df[self.fit_feats])
        return df

    def save(self, name):
        path = PROCESSED_DIR / self.project / 'models'
        if not os.path.exists(path):
            os.mkdir(path)

        payload = {'omit': self.omit_cols,
                   'means': self.col_means,
                   'feats': self.fit_feats,
                   'scaler': self.scaler}

        with open(path / f'{name}_processor.pkl', 'wb') as f:
            pkl.dump(payload, f)

    def load(self, name):
        path = PROCESSED_DIR / self.project / 'models'
        with open(path / f'{name}_processor.pkl', 'rb') as f:
            payload = pkl.load(f)
            self.omit_cols = payload['omit']
            self.col_means = payload['means']
            self.fit_feats = payload['feats']
            self.scaler = payload['scaler']
