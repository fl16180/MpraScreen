import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


mpra_base = "/oak/stanford/groups/zihuai/fredlu/processed/genomeScreen2/fragments/"


def compile_scores():
    # Pre-defined range to look at (in this case 1.5 million)
    start_pos = 65010000
    end_pos = 80010000

    # Takes advantage of fact every fragment is 25000 base pairs long
    positions = np.arange(start_pos, end_pos, 25000)
    fname = ""

    fragment_files = os.listdir(mpra_base)

    scores = []
    pos = []
    for x in range(len(positions)-1):
        starting_pos = positions[x]

        # Not maximum efficient but what i have been using - to find correct fragment file that is next
        for i in fragment_files:
            if "start_" + str(starting_pos) in i:
                fname = mpra_base + i
                break
    
        # For counting progress, found fname file and path
        # print(fname)

        df = pd.read_csv(fname)
        scores.append(df['MPRA_raw'].values)
        pos.append(df['pos'].values)


    scores = np.concatenate(scores)
    pos = np.concatenate(pos)

    np.save('./score_segment.npy', scores)
    np.save('./pos_segment.npy', pos)


def visualize():
    scores = np.load('./score_segment.npy')
    pos = np.load('./pos_segment.npy')

    q95 = np.percentile(scores, 95)
    print(q95)
    high_scores = scores[scores > q95]

    bins = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    scoremap = np.zeros((len(bins), len(scores)))

    for b in bins:
        filt = np.ones(b)
        
        np.convolve(high_scores, filt, mode='same')


if __name__ == '__main__':

    compile_scores()
    # visualize()

    # Saving scores from fragment files
    # mpra_scores.update(tabix.retrieve_mpra_output_csv(fname,make_rank_column=False))