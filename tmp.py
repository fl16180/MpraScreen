import pandas as pd

from genome_wide_screen import MpraScreen
from datasets import load_full_project_data
from constants import PROCESSED_DIR

import os
import pandas as pd
import numpy as np
import io, subprocess
from tqdm import tqdm
# from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder

from utils.bigwig_utils import pull_roadmap_features, compile_roadmap_features
from constants import *
from pathlib import Path


def make_output_csv(project, split, raw_score, calib_score):
    df = pd.read_csv(PROCESSED_DIR / project / f'matrix_{split}.csv',
                     usecols=[0, 1, 2])
    ref = df[['chr', 'pos', 'Label']].copy()
    ref['raw_score'] = raw_score
    ref['calib_score'] = calib_score
    ref.to_csv(
        PROCESSED_DIR / 'analysis' / f'{project}_{split}_scores.csv',
        index=False
    )


def get_random_bed(n_samples=100000):
    """ Generate random bedfile corresponding to n_samples random locations
    in the genome. Used for picking locations to extract unlabeled
    data for semi-supervised learning.

    Offset reduces the range of indices per chromosome to sample, for example
    when edges are not available.
    """
    total_range = sum([x for x in GenRange.values()])

    chrs = []
    samples = []
    for chrom in range(1, 23):
        select = int(n_samples * GenRange[chrom] / total_range)
        bps = np.random.randint(low=0, high=GenRange[chrom], size=select)
        samples.append(bps)
        chrs.extend([chrom] * select)

    samples = np.concatenate(samples)
    chrs = np.array(chrs)

    bed = pd.DataFrame({'chr': chrs,
                        'pos': samples,
                        'pos_end': samples+1,
                        'rs': ['ul{0}'.format(x+1) for x in range(len(samples))]})
    return bed


def extract_mpra_screen(bedfile, outpath):
    """ Extract MPRA Screen predictions using an input bed table.
        
        The bed file should be a pandas DataFrame with the following format:
            chr (int): chromosome
            pos (int): position of interest
            variant (str): (optional) name or rsid assigned to variant for naming 

        The extracted data is saved as a tsv to the specified outpath.
    
        Note: currently only supports hg19 scores and extracts one position per row.
    """
    if os.path.exists(outpath):
        os.remove(outpath)

    SCORE_DIR = Path('/oak/stanford/groups/zihuai/fredlu/processed/genomeScreen/scores')
    SCORE_BASE = 'mpra_screen_chr_XX.tsv.gz'
    
    # load column header names
    head_file = SCORE_DIR / 'header.txt'
    with open(head_file, 'r') as f:
        header = f.read()
    header = header.strip('\n').split('\t')

    with open(outpath, 'w') as fp:
        fp.write('\t'.join(header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            # make system command
            command = f'tabix {SCORE_DIR}/{SCORE_BASE} '
            args = f'{row.chr}:{row.pos}-{row.pos}'
            full_cmd = command.replace('XX', str(row.chr)) + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                # read query string
                query = io.StringIO(stdout.decode('utf-8'))
                features = pd.read_csv(query, sep=r'\s+', header=None,
                                       index_col=False, na_values='.')

                # remove tabix end-position overlapping with query position
                features = features[features[1] == row.pos]

                features = features.astype(str).values.tolist()
                for fr in features:
                    fp.write('\t'.join(fr) + '\n')


if __name__ == '__main__':

    # screen = MpraScreen()

    # # mpra_nova + 1kg_background train set
    # print('nova train')
    # X, X_neighbor = load_full_project_data('final_1kg', split='train')
    # X = screen.preprocess(X, X_neighbor)
    # raw_score = screen.predict(X)
    # calib_score = screen.predict_calibrate(X)
    # make_output_csv('final_1kg', 'train', raw_score, calib_score)

    # # mpra_nova + 1kg_background test set
    # print('nova test')
    # X, X_neighbor = load_full_project_data('final_1kg', split='test')
    # X = screen.preprocess(X, X_neighbor)
    # raw_score = screen.predict(X)
    # calib_score = screen.predict_calibrate(X)
    # make_output_csv('final_1kg', 'test', raw_score, calib_score)

    # # random 1kg_background set
    # print('background all')
    # X, X_neighbor = load_full_project_data('1kg_background4', split='all')
    # X = screen.preprocess(X, X_neighbor)
    # raw_score = screen.predict(X)
    # calib_score = screen.predict_calibrate(X)
    # make_output_csv('1kg_background4', 'all', raw_score, calib_score)

    # from genome_wide_screen import sample_for_calibrator

    # sample_for_calibrator(100000) 
    import numpy as np
    import matplotlib.pyplot as plt

    from utils.bed_utils import get_random_bed
    
    bed = get_random_bed(n_samples=20000)
    print(bed.head())

    extract_mpra_screen(bed, './tmp.tsv')

    # sample_vals = []
    # for chrom in range(1, 23):
    #     df = pd.read_csv(PROCESSED_DIR / 'genomeScreen' / f'tmp_{chrom}.csv', sep='\t')
    #     sample_vals.append(df.dropna().iloc[:, 4].values)

    # df = pd.read_csv('./tmp.tsv', sep='\t')

    # all_sample_vals = np.concatenate(sample_vals)
    # print(all_sample_vals.shape)
    
    # plt.hist(df.MPRA_raw, bins=20)
    # plt.savefig('./raw_score_hist.png')
    # plt.close()

    # plt.hist(df.MPRA_score, bins=20)
    # plt.savefig('./calibrated_score_hist.png')
    # plt.close()

    # plt.hist(df.MPRA_PHRED, bins=20)
    # plt.savefig('./phred_score_hist.png')
    # plt.close()
