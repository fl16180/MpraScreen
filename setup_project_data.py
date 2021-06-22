import argparse
import os
import pandas as pd

from constants import *
from utils.bed_utils import *
from utils.data_utils import *
from datasets import load_data_set

SPLIT_CHOICES = [None, 'train-test', 'all']


def setup(args):

    project_dir = PROCESSED_DIR / args.project

    # load or create variants bedfile
    try:
        if args.bed:
            raise Exception('Overwriting bedfile')
        bedfile = load_bed_file(args.project)
        print(f'Loaded bedfile from {args.project}')
    except Exception as e:
        if 'unif_background' in args.project:
            bedfile = get_random_bed(n_samples=100000)
        elif '1kg_background' in args.project:
            bedfile = sample_1kg_background(n_samples=100000)
        else:
            bedfile = get_bed_from_mpra(args.project)
        save_bed_file(bedfile, args.project)
        print(f'Generated new bedfile in {args.project}')

    if args.roadmap:
        print('Extracting Roadmap: ')
        fname = project_dir / 'roadmap_extract.tsv'
        extract_roadmap(bedfile.copy(), fname, args.project)

    if args.regbase:
        print('Extracting regBase: ')
        fname = project_dir / 'regBase_extract.tsv'
        extract_regbase(bedfile.copy(), fname)

    if args.eigen:
        print('Extracting Eigen: ')
        fname = project_dir / 'eigen_extract.tsv'
        extract_eigen(bedfile.copy(), fname)

    if args.genonet:
        print('Extracting GenoNet: ')
        fname = project_dir / f'GenoNet_extract.tsv'
        extract_genonet(bedfile.copy(), fname, 'mean')

    # further process data and split into train/test sets
    if args.split == 'train-test':
        bed_train, bed_test = split_train_test(bedfile,
                                               test_frac=0.25,
                                               seed=args.seed)

        process_datasets(args, bed_train, split='train')
        process_datasets(args, bed_test, split='test')

    elif args.split == 'all':
        process_datasets(args, bedfile, split='all')


def process_datasets(args, bedfile, split='all', merge='inner'):
    print(f'Processing {split} set')
    project_dir = PROCESSED_DIR / args.project

    if 'background' in args.project or (args.project == 'genomeScreen' or 'gs_job' in args.project):
        mpra = bedfile.copy()
        mpra['Label'] = 0
    else:
        mpra = load_mpra_data(args.project)

    if args.project == 'mpra_nova':
        mpra['Label'] = mpra['Hit']

    y_split = pd.merge(bedfile, mpra, on=['chr', 'pos'], how=merge).loc[:, ['chr', 'pos', 'Label']]
    y_split.to_csv(project_dir / f'{split}_label.csv', sep=',', index=False)

    if os.path.exists(project_dir / 'roadmap_extract.tsv'):
        print('\tRoadmap')
        roadmap = pd.read_csv(project_dir / 'roadmap_extract.tsv', sep='\t')
        try:
            roadmap['chr'] = roadmap['chr'].map(lambda x: x[3:])
        except TypeError:
            pass
        roadmap[['chr', 'pos']] = roadmap[['chr', 'pos']].astype(int)
        r_split = pd.merge(bedfile[['chr', 'pos']], roadmap, on=['chr', 'pos'], how=merge)
        r_split.to_csv(project_dir / f'{split}_roadmap.csv', sep=',', index=False)

    if os.path.exists(project_dir / 'regBase_extract.tsv'):
        print('\tregBase')
        regbase = clean_regbase_data(project_dir / 'regBase_extract.tsv')
        r_split = pd.merge(bedfile[['chr', 'pos']], regbase, how=merge)
        r_split.to_csv(project_dir / f'{split}_regbase.csv', index=False)

    if os.path.exists(project_dir / 'eigen_extract.tsv'):
        print('\tEigen')
        eigen = clean_eigen_data(project_dir / 'eigen_extract.tsv')
        e_split = pd.merge(bedfile[['chr', 'pos']], eigen, how=merge)
        e_split.to_csv(project_dir / f'{split}_eigen.csv', index=False)

    # compile data splits into single csv
    _ = load_data_set(args.project, split=split, make_new=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', choices=PROJ_CHOICES, required=True)
    parser.add_argument('--bed', '-b', default=False, action='store_true',
                        help='(re-)extract variant bed from target data')
    parser.add_argument('--roadmap', '-r', default=False, action='store_true',
                        help='extract Roadmap data')
    parser.add_argument('--regbase', '-rb', default=False, action='store_true',
                        help='extract regBase data')
    parser.add_argument('--eigen', '-e', default=False, action='store_true',
                        help='extract Eigen data')
    parser.add_argument('--genonet', '-g', default=False, action='store_true',
                        help='extract GenoNet data')
    parser.add_argument('--split', '-s', default=None, choices=SPLIT_CHOICES,
                        help='split data into train/test sets or all')
    parser.add_argument('--seed', default=9999, help='train/test random seed')
    args = parser.parse_args()

    setup(args)
