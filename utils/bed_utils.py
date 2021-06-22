import numpy as np
import pandas as pd
from utils.data_utils import load_mpra_data
from constants import *


def load_bed_file(project, name=None, chrXX=False,
                  extra_cols=[]):
    if not name:
        name = project
    path = PROCESSED_DIR / f'{project}/{name}.bed'
    bed = pd.read_csv(path, sep='\t', header=None,
                      names=['chr', 'pos', 'pos_end', 'rs'] + extra_cols)
    if chrXX:
        re_filt = bed['chr'].str.contains('chr\d{1,2}$')
        bed = bed[re_filt]
        bed['chr'] = bed['chr'].map(lambda x: x[3:])

    bed[['chr', 'pos', 'pos_end']] = bed[['chr', 'pos', 'pos_end']].astype(int)
    return bed


def save_bed_file(bedfile, project):
    path = PROCESSED_DIR / f'{project}/{project}.bed'
    bedfile.to_csv(path, sep='\t', index=False, header=False)


def get_bed_from_mpra(dataset):
    """ The raw MPRA data files contain non-coding variant locations already
    merged with ROADMAP epigenetic data. This function extracts the variant
    locations and converts to bedfile format.
    """
    data = load_mpra_data(dataset)
    bed = data[['chr', 'pos', 'rs']].copy()

    bed['pos_end'] = bed['pos'] + 1
    bed = bed[['chr', 'pos', 'pos_end', 'rs']]
    return bed


def get_random_bed(n_samples=100000, offset=0):
    """ Generate random bedfile corresponding to n_samples random locations
    in the genome. Used for picking locations to extract unlabeled
    data for semi-supervised learning.

    Offset reduces the range of indices per chromosome to sample, for example
    when edges are not available.
    """
    total_range = sum([x for x in GenRange.values()])

    chrs = []
    samples = []
    for chr in range(1, 23):
        select = int(n_samples * GenRange[chr] / total_range)
        bps = np.random.randint(low=0, high=GenRange[chr] - offset, size=select)
        samples.append(bps)
        chrs.extend([chr] * select)

    samples = np.concatenate(samples)
    chrs = np.array(chrs)

    bed = pd.DataFrame({'chr': chrs,
                        'pos': samples,
                        'pos_end': samples+1,
                        'rs': ['ul{0}'.format(x+1) for x in range(len(samples))]})
    return bed


def sample_1kg_background(n_samples=100000):
    df = pd.read_csv(BACKGROUND_1KG, sep='\t')
    df = df.sample(n=n_samples, replace=False)

    df['chr'] = df['chr'].map(lambda x: x[3:])
    df[['chr', 'pos']] = df[['chr', 'pos']].astype(int)
    df['pos_end'] = df['pos'] + 1
    bed = df[['chr', 'pos', 'pos_end', 'rs']]
    return bed
