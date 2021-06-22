import os
import csv
import argparse
import numpy as np
import pandas as pd
import shutil
import pickle

from utils.model_utils import load_model, load_calibrator
from utils.data_utils import extract_roadmap, extract_regbase, extract_eigen, get_roadmap_col_order
from utils.neighbor_utils import add_bed_neighbors, pull_roadmap_features, \
    reshape_roadmap_files, process_roadmap_neighbors
from datasets import load_data_set, Processor
from constants import PROCESSED_DIR, TMP_DIR, ROADMAP_MARKERS, EPS, GenRange
from setup_project_data import process_datasets
from models import Calibrator


class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class RollingDataWindow:

    def __init__(self, chrom, width=5000, start=1000, end=100000, buffer=1000, job_id=0):
        self.chrom = chrom
        self.width = width
        self.start = start
        self.end = end
        self.buf = buffer
        self.left_buffer = self.right_buffer = None
        self.proj = f'gs_job_{job_id}_chr_{chrom}'

        assert self.start - self.buf >= 0
        self.left = self.right = 0

    def setup_window(self):
        if self.left_buffer is None or self.right_buffer is None:
            self.left = self.start
            self.right = min(self.start + self.width, self.end)
            self.bed = self.make_bed_window(self.chrom,
                                            start=self.left,
                                            end=self.right)

            initial_bed = self.make_bed_window(self.chrom,
                                               start=self.left - self.buf,
                                               end=self.right + self.buf)
            
            self.extract_data(initial_bed)
            df = load_data_set(self.proj, split='all', make_new=False)

            self.df = df.iloc[self.buf:-self.buf, :]
            self.left_buffer = df.iloc[:self.buf, :]
            self.right_buffer = df.iloc[-self.buf:, :]

        else:
            self.left = self.right
            self.right = min(self.right + self.width, self.end)
            self.bed = self.make_bed_window(self.chrom,
                                            start=self.left,
                                            end=self.right)

            new_bed = self.make_bed_window(self.chrom,
                                           start=self.left + self.buf,
                                           end=self.right + self.buf)
            self.extract_data(new_bed)
            df = load_data_set(self.proj, split='all', make_new=False)

            self.left_buffer = self.df.iloc[-self.buf:, :]
            self.df = pd.concat([self.right_buffer, df.iloc[:-self.buf]], axis=0)
            self.right_buffer = df.iloc[-self.buf:, :]

    def extract_data(self, bed):
        ''' extract SNV-site annotations and scores '''
        project_dir = PROCESSED_DIR / self.proj

        fname = project_dir / 'roadmap_extract.tsv'
        extract_roadmap(bed.copy(), fname, self.proj)

        fname = project_dir / 'regBase_extract.tsv'
        extract_regbase(bed.copy(), fname)

        fname = project_dir / 'eigen_extract.tsv'
        extract_eigen(bed.copy(), fname)

        args = MockArgs(project=self.proj)
        process_datasets(args, bed, split='all', merge='left')

    def add_neighbors_from_buffer(self):
        NEIGHBOR_COLS = [f'{x}-E116' for x in ROADMAP_MARKERS]

        n_neigh, sample_res = 40, 25
        self.bed['rs'] = self.bed['chr'].map(str) + '-' + self.bed['pos'].map(str)
        self.bed.drop_duplicates('rs', inplace=True)

        neighbor_bed = add_bed_neighbors(self.bed, n_neigh, sample_res)

        all_data = pd.concat([self.left_buffer, self.df, self.right_buffer], axis=0)
        joined_df = pd.merge(neighbor_bed,
                             all_data[['chr', 'pos'] + NEIGHBOR_COLS],
                             on=['chr', 'pos'],
                             how='left')
        
        arr_out = np.zeros((self.bed.shape[0], 8, 81))
        joined_df[['chr-pos', 'neighbor']] = joined_df['rs'].str.split(';', n=1, expand=True)
        joined_df['neighbor'] = joined_df['neighbor'].astype(int)
        for i, feat in enumerate(NEIGHBOR_COLS):
            tmp = joined_df.pivot(index='chr-pos', values=feat, columns='neighbor')
            tmp = tmp.loc[:, np.sort(tmp.columns)]
            arr_out[:, i, :] = tmp.values
            
        self.neighbors = arr_out

    def make_bed_window(self, chrom, start, end):
        pos = np.arange(start, end)
        pos_end = pos + 1
        bed = pd.DataFrame({'pos': pos, 'pos_end': pos_end})
        bed['chr'] = int(chrom)
        bed['rs'] = [f'tmp{x}' for x in range(bed.shape[0])]
        return bed[['chr', 'pos', 'pos_end', 'rs']]

    def in_progress(self):
        return self.right < self.end

    def get_data(self):
        return self.df, self.neighbors, self.bed


class MpraScreen:
    def __init__(self):
        self.roadmap_cols = get_roadmap_col_order(order='marker')
        self.preprocessor = Processor('final_1kg')
        self.preprocessor.load('neighbors')
        self.net = load_model('final_1kg', 'neighbors')
        self.calibrator = load_calibrator('final_1kg', 'neighbors')

    def preprocess(self, X, X_neighbor):
        X[self.roadmap_cols] = np.log(X[self.roadmap_cols] + EPS)

        X = self.preprocessor.transform(X)

        rm_cols = [f'{x}-E116' for x in ROADMAP_MARKERS]
        X_score = X.drop(['chr', 'pos', 'Label'] + rm_cols, axis=1) \
                   .values \
                   .astype(np.float32)
        
        X_neighbor = np.log(X_neighbor.astype(np.float32) + EPS)
        X_neighbor = X_neighbor.reshape(
            X_neighbor.shape[0], X_neighbor.shape[1] * X_neighbor.shape[2]
        )
        X = np.hstack((X_score, X_neighbor))
        return X

    def predict(self, X):
        return self.net.predict_proba(X)[:, 1]

    def predict_calibrate(self, X):
        scores = self.net.predict_proba(X)[:, 1]
        return self.calibrator.transform(scores)

    def predict_phred(self, X):
        scores = self.predict_calibrate(X)
        return -10 * np.log10(1 - scores)


def join_fragments(chrom):
    fragment_dir = PROCESSED_DIR / 'genomeScreen' / 'fragments'
    fragments = os.listdir(fragment_dir)

    # collect all score fragments for chrom
    fragments = [x for x in fragments if f'_chr_{chrom}_' in x]
    starts = [x.split('start_')[1].split('_end')[0] for x in fragments]
    ends = [x.split('end_')[1].split('.csv')[0] for x in fragments]

    # compile in dataframe
    frag_table = pd.DataFrame({'file': fragments, 'start': starts, 'end': ends})
    frag_table[['start', 'end']] = frag_table[['start', 'end']].astype(int)
    frag_table = frag_table.sort_values('start').reset_index(drop=True)

    print(frag_table.head())
    print(frag_table.tail())

    if frag_table['start'].iloc[0] == 10000:
        print('Starting value met')
    if frag_table['end'].iloc[-1] == GenRange[chrom] - 10000:
        print('Ending value met')

    print('Validating...')
    gaps = []
    for i in range(1, frag_table.shape[0]):
        next_start = frag_table.loc[i, 'start']
        prev_end = frag_table.loc[i - 1, 'end']
        assert next_start >= prev_end
        if next_start > prev_end:
            print('Gap detected:', prev_end, next_start)
            gaps.append((prev_end, next_start))

    if frag_table['start'].iloc[0] != 10000:
        gaps.append((10000, frag_table['start'].iloc[0]))
    if frag_table['end'].iloc[-1] != GenRange[chrom] - 10000:
        gaps.append((frag_table['end'].iloc[-1], GenRange[chrom] - 10000))
    gaps = pd.DataFrame(np.array(gaps))
    gaps.to_csv(PROCESSED_DIR / 'genomeScreen' / f'gaps_chr_{chrom}.csv', index=False)

    if len(gaps) <= 1:
        print('Loading...')
        dfs = [pd.read_csv(fragment_dir / x) for x in frag_table['file'].values]

        # dfs = []
        # idx = 0
        # while True:
        #     # if idx % 100 == 1:
        #         # print(idx)
        #     start = frag_table.loc[idx, 'start']
        #     end = frag_table.loc[idx, 'end']
        #     loc = frag_table.loc[idx, 'file']

        #     dfs.append(pd.read_csv(fragment_dir / loc))

        #     idx = frag_table.index[frag_table['start'] == end].tolist()
        #     if len(idx) >= 1:
        #         idx = idx[0]
        #     else:
        #         break

        print('Stacking...')
        chrom_scores = pd.concat(dfs, axis=0)
        print('Saving...')
        chrom_scores.to_csv(PROCESSED_DIR / 'genomeScreen' / f'mpra_screen_chr_{chrom}.csv', index=False, sep='\t')


def validate_chr_scores(chrom):
    with open(PROCESSED_DIR / 'genomeScreen' / f'mpra_screen_chr_{chrom}.csv') as f:
        # reader = csv.reader(f, delimiter='\t')
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        pos = int(next(reader)[1])
        print(f'Starting: {pos}')

        for row in reader:
            if pos % 5000000 == 1:
                print(f'Now at {pos}')
            if int(row[1]) != pos + 1:
                print(f'Gap: {pos}-{int(row[1])}')
            pos = int(row[1])
    
    print(f'Ending: {pos}')


def sample_for_calibrator(n_samples=5000000):
    from utils.bed_utils import get_random_bed
    from subprocess import call

    bed = get_random_bed(n_samples=n_samples)
    print(bed.shape)
    for chrom in range(1, 23):
        n_sample = bed[bed.chr == chrom].shape[0]
        print(chrom, n_sample)

        command = f'shuf -n {n_sample} ../processed/genomeScreen/mpra_screen_chr_{chrom}.csv > ../processed/genomeScreen/tmp_{chrom}.csv'
        call(command, shell=True)


def learn_calibrator():

    sample_vals = []
    for chrom in range(1, 23):
        df = pd.read_csv(PROCESSED_DIR / 'genomeScreen' / f'tmp_{chrom}.csv', sep='\t')
        sample_vals.append(df.dropna().iloc[:, 4].values)

    all_sample_vals = np.concatenate(sample_vals)
    print(all_sample_vals.shape)

    cal = Calibrator()
    cal.fit(all_sample_vals)

    fname = PROCESSED_DIR / 'genomeScreen' / 'background_calibrator.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(cal, f)


def add_calibrations(chrom):
    # import dask.dataframe as dd

    fname = PROCESSED_DIR / 'genomeScreen' / 'background_calibrator.pkl'
    with open(fname, 'rb') as f:
        cal = pickle.load(f)

    # for chrom in range(1, 23):
    print(f'Loading {chrom}')
    df = pd.read_csv(PROCESSED_DIR / 'genomeScreen' / f'mpra_screen_chr_{chrom}.csv', sep='\t')
    df.drop('rs', axis=1, inplace=True)
    df.dropna(inplace=True)
    df.rename(columns={'pos': 'pos_start'}, inplace=True)
    df[['chr', 'pos_start', 'pos_end']] = df[['chr', 'pos_start', 'pos_end']].astype(int)

    print('Transforming')
    df['MPRA_score'] = cal.transform(df['MPRA_raw'].values)
    df['MPRA_PHRED'] = -10 * np.log10(1 - df['MPRA_score'] + 1e-8)
    
    print('Saving')
    df.to_csv(PROCESSED_DIR / 'genomeScreen' / 'scores' / f'mpra_screen_chr_{chrom}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=int, help='cluster job ID', default=0)
    parser.add_argument('--chr', type=int, help='chromosome to run')
    parser.add_argument('--width', type=int, default=10000, help='batch window width')
    parser.add_argument('--start', type=int, default=1000, help='start index, inclusive')
    parser.add_argument('--end', type=int, default=10000, help='end index, not inclusive')
    parser.add_argument('--concat', action='store_true', default=False, help='Concat fragments')
    parser.add_argument('--calibrate', action='store_true', default=False)
    args = parser.parse_args()

    if not args.concat:
        # create job storage directory
        if not os.path.exists(PROCESSED_DIR / f'gs_job_{args.job}_chr_{args.chr}'):
            os.makedirs(PROCESSED_DIR / f'gs_job_{args.job}_chr_{args.chr}')

        # load complete model
        screen = MpraScreen()

        # set up window data generator
        node = RollingDataWindow(args.chr,
                                width=args.width,
                                start=args.start,
                                end=args.end,
                                buffer=1000,
                                job_id=args.job)
        
        node_pred = pd.DataFrame()
        while node.in_progress():
            node.setup_window()
            node.add_neighbors_from_buffer()

            X_site, X_neighbors, bed = node.get_data()
            X = screen.preprocess(X_site, X_neighbors)
            bed['MPRA_raw'] = screen.predict(X)
            # bed['MPRA_screen'] = screen.predict_calibrate(X)
            # bed['MPRA_PHRED'] = screen.predict_phred(X)
            node_pred = pd.concat([node_pred, bed], axis=0)

        node_pred.to_csv(
            PROCESSED_DIR / 'genomeScreen' / 'fragments' / f'node_{args.job}_chr_{args.chr}_start_{args.start}_end_{args.end}.csv',
            index=False
        )

        shutil.rmtree(PROCESSED_DIR / 'tmp' / f'gs_job_{args.job}_chr_{args.chr}')
    
    else:
        if not args.calibrate:
            # python genome_wide_screen.py --chr 3 --concat
            join_fragments(args.chr)

        else:
            # python genome_wide_screen.py --concat --calibrate

            # sample_for_calibrator()
            # learn_calibrator()
            add_calibrations(args.chr)

