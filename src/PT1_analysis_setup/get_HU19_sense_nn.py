import os
import sys
import re
import csv
import json
import shutil
import pickle
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial import distance

COHA_SENSE_PROPS_DIR = './data/COHA/hidden{}-masked{}/hu19_props'
HU19_DIR = './data/hu2019'


def get_init_sense_props(coha_sense_props_dir):
    """ Return a dictionary mapping each sense label to its initial frequency
    mapping. 
    """
    init_sense_props = {}

    for f_name in tqdm(os.listdir(coha_sense_props_dir)):
        df = pd.read_csv(os.path.join(coha_sense_props_dir, f_name))
        df_sorted = df.sort_values(by=['hu19_sense', 'year'])
        sense_props = df_sorted[['hu19_sense', 'prop_lexicon']].groupby('hu19_sense')['prop_lexicon'].apply(list).to_dict()  # https://stackoverflow.com/a/50505848

        for sense in sense_props:
            init_sense_props[sense] = sense_props[sense][0]
    
    return init_sense_props

def get_HU19_sense_nn():
    """ Save, in increasing order, the list of nearest senses in semantic space
    for the senses used in Hu et al. (2019). We only include senses whose 
    intial frequency is greater than 1/100K. 
    """
    coha_sense_props_dir = COHA_SENSE_PROPS_DIR.format(1, 'N')
    hu19_embeds_path = os.path.join(HU19_DIR, 'diachronic_sense_emb.pkl')
    hu19_nn_path = os.path.join(HU19_DIR, 'sense_nn.pkl')
    
    print('Loading data...')
    init_sense_props = get_init_sense_props(coha_sense_props_dir)
    with open(hu19_embeds_path, 'rb') as in_f:
        hu19_embeds = pickle.load(in_f)

    print('Removing under 1/100K init freq senses...')
    print(f'Initial: {len(hu19_embeds)}')
    for sense in list(hu19_embeds.keys()):
        if sense not in init_sense_props or init_sense_props[sense] < 0.000001:
            del hu19_embeds[sense]
    print(f'Final: {len(hu19_embeds)}')

    print('Obtaining NN of HU19 senses...')
    hu19_senses = np.array(list(hu19_embeds.keys()))
    embeds = np.array([hu19_embeds[sense] for sense in hu19_embeds])
    nn = {}  # sense --> [nearest senses to farthest senses]
    
    for i in tqdm(range(len(hu19_senses))):
        cur_dists = []
        for j in range(len(hu19_senses)):
            cur_dists.append(distance.cosine(embeds[i], embeds[j]))
        cur_idxs = np.argsort(cur_dists)[1:]
        
        cur_sense = hu19_senses[i]
        nn[cur_sense] = hu19_senses[cur_idxs]
    
    print('Saving...')
    with open(hu19_nn_path, 'wb') as out_f:
        pickle.dump(nn, out_f)

if __name__ == "__main__":
    get_HU19_sense_nn()
