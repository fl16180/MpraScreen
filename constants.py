from pathlib import Path


HOME_DIR = Path('/oak/stanford/groups/zihuai/fredlu')
EIGEN_DIR = Path('/oak/stanford/groups/zihuai/FST/Score')
ROADMAP_DIR = Path('/oak/stanford/groups/zihuai/SemiSupervise/bigwig/rollmean')
BBANK_DIR = HOME_DIR / 'BioBank'
REGBASE_DIR = HOME_DIR / 'regBase' / 'V1.1'
GENONET_DIR = Path('/oak/stanford/groups/zihuai/GenoNet/GenoNetScores_byChr')
EXPECTO_DIR = HOME_DIR / 'exPecto'

MPRA_DIR = HOME_DIR / 'MPRA'
CODE_DIR = HOME_DIR / 'MpraScreen'
PROCESSED_DIR = HOME_DIR / 'processed'
TMP_DIR = HOME_DIR / 'processed' / 'tmp'

REGBASE = 'regBase_V1.1.gz'
EIGEN_BASE = 'Eigen_hg19_noncoding_annot_chrXX.tab.bgz'
GENONET_BASE = 'GenoNet_XX.bed.gz'

MPRA_TABLE = {
    'mpra_e116': ('LabelData_CellPaperE116.txt', 'TestData_MPRA_E116_unbalanced.txt'),
    'mpra_e118': ('LabelData_KellisE118.txt', 'TestData_MPRA_E118.txt'),
    'mpra_e123': ('LabelData_KellisE123.txt', 'TestData_MPRA_E123.txt'),
    'mpra_nova': ('1KG_bartender_novaSeq_DESeq2-LoveInteract-sumstats.txt', '')
}

# project choices for all argparsers
PROJ_CHOICES = ['mpra_e116', 'mpra_e118', 'mpra_e123', 'mpra_nova',
                'gnom_mpra_mixed', 'big_random_set']

# experiments with ROADMAP data included
STANDARD_MPRA = ('mpra_e116', 'mpra_e118', 'mpra_e123')

BIGWIG_UTIL = '/home/users/fredlu/opt/bigWigAverageOverBed'
BIGWIG_TAIL = '.imputed.pval.signal.bigwig'
ROADMAP_MARKERS = ['DNase', 'H3K27ac', 'H3K27me3', 'H3K36me3',
                   'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']
ROADMAP_COL_ORDER_REF = 'mpra_e116'

N_NEIGH = 40
SAMPLE_RES = 25

# epsilon offset for log of roadmap 
EPS = 1e-4

# file for sampling 1K Genome locations
BACKGROUND_1KG = '/oak/stanford/groups/zihuai/NGreview/Mac5Eur/valley9.c89_Mac5Eur_E126_predictions.txt.gz'

# hg19 chromosome lengths
GenRange = {1: 249250621,
            2: 243199373,
            3: 198022430,
            4: 191154276,
            5: 180915260,
            6: 171115067,
            7: 159138663,
            8: 146364022,
            9: 141213431,
            10: 135534747,
            11: 135006516,
            12: 133851895,
            13: 115169878,
            14: 107349540,
            15: 102531392,
            16: 90354753,
            17: 81195210,
            18: 78077248,
            19: 59128983,
            20: 63025520,
            21: 48129895,
            22: 51304566
}
