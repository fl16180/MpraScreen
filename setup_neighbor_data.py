import argparse

from constants import PROCESSED_DIR, PROJ_CHOICES, TMP_DIR
from utils.bed_utils import load_bed_file
from utils.bigwig_utils import pull_roadmap_features
from utils.neighbor_utils import reshape_roadmap_files, add_bed_neighbors, process_roadmap_neighbors
from datasets.data_loader import *

SPLIT_CHOICES = [None, 'train-test', 'all']


def setup(args):
    n_neigh, sample_res = map(int, args.neighbor_param.split(','))

    project_dir = PROCESSED_DIR / args.project
    bedfile = load_bed_file(args.project)

    # assign new variant name for data merging later
    bedfile['rs'] = bedfile['chr'].map(str) + '-' + bedfile['pos'].map(str)
    bedfile.drop_duplicates('rs', inplace=True)

    # neighbors directory
    neighbor_loc = TMP_DIR / f'{args.project}_neighbor'

    # extract roadmap data from bigwig files
    if args.extract:
        neighbor_bed = add_bed_neighbors(bedfile, n_neigh, sample_res)
        pull_roadmap_features(neighbor_bed, feature_dir=neighbor_loc)
        reshape_roadmap_files(feature_dir=neighbor_loc)

    # prepare roadmap data into train and test sets corresponding to train/test
    # dataframes generated from other datasets
    if args.split == 'train-test':
        train_df = load_data_set(args.project, split='train',
                                 make_new=True)

        test_df = load_data_set(args.project, split='test',
                                make_new=True)
        
        outpath1 = f'train_{n_neigh}_{sample_res}'
        outpath2 = f'test_{n_neigh}_{sample_res}'
        process_roadmap_neighbors([train_df, test_df], [outpath1, outpath2],
                                  project_dir, args.tissue,
                                  neighbor_loc)

    elif args.split == 'all':
        test_df = load_data_set(args.project, split='all',
                                make_new=True)

        outpath = f'all_{n_neigh}_{sample_res}'
        process_roadmap_neighbors([test_df], [outpath],
                                  project_dir, args.tissue,
                                  neighbor_loc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', required=True)
    parser.add_argument('--extract', '-e', default=False, action='store_true',
                        help='extract neighboring Roadmap data')
    parser.add_argument('--tissue', '-t', default='all', type=str,
                        help='get neighbor data for specific tissue e.g. E116')
    parser.add_argument('--split', '-s', default=None, choices=SPLIT_CHOICES,
                        help='split data into train/test sets or all')
    parser.add_argument('--neighbor_param', '-npr', default='40,25', type=str,
                        help='Roadmap neighbor params: (n_neigh,sample_res)')
    args = parser.parse_args()

    setup(args)
