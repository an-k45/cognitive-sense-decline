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
from scipy.stats import spearmanr

COHA_SENSE_PROPS_DIR = './data/COHA/hidden1-maskedN/hu19_props'
COHA_SENSE_ANALYSIS_DIR = './data/COHA/hidden1-maskedN/sense_analysis_all10nn_matnanmean'
COHA_SENSE_NN_EX_DIR = './data/COHA/hidden1-maskedN/sense_factor_nn_ex'
HU19_DIR = './data/hu2019'
KUPER13_DIR = './data/kuperman13'

# FACTOR EVAL FUNCS, ADAPTED FROM MAIN ANALYSIS

def eval_concreteness(cur_sense, sense_nn, conc_ratings):
    """ Given a sense, return the weighted mean of the concreteness ratings
    for 10NN, using word concreteness ratings by Kuperman 2013.
    Return NaN if 6 or less neighbours have a rating.
    """
    concs = []
    for i in range(10):
        sense = sense_nn[cur_sense][i]
        word = sense.split('_')[0]

        if word in conc_ratings:
            concs.append(conc_ratings[word])
        else:
            concs.append(np.nan)

    arr = np.array(concs) - 3
    return np.nanmean(arr), arr

def eval_valence(cur_sense, sense_nn, val_ratings):
    """ Given a sense, return the weighted mean of the valence ratings for 10NN, 
    using word concreteness ratings by Kuperman 2013.
    Return -10 if 6 or less neighbours have a rating.
    """
    vals = []
    for i in range(10):
        sense = sense_nn[cur_sense][i]
        word = sense.split('_')[0]

        if word in val_ratings:
            vals.append(val_ratings[word])
        else:
            vals.append(np.nan)
    
    arr = np.abs(np.array(vals) - 5)
    return np.nanmean(arr), arr

def eval_demand(cur_sense, sense_nn):
    """ Given a sense, for its 10NN, compute the mean value of the Spearman
    correlation between {1,2,...,20} and frequencies from t=1,2,...,20

    Omit senses which have a decline fit less than 0.1. 
    """
    comparison = list(range(1,21))

    corrs = []
    for i in range(10):
        sense = sense_nn[cur_sense][i]
        word = sense.split('_')[0]
        df_word_props = pd.read_csv(os.path.join(COHA_SENSE_PROPS_DIR, f'{word}.csv'))

        freqs = df_word_props[df_word_props['hu19_sense'] == sense]['prop_lexicon'].to_list()

        if len(freqs) != 20:
            corrs.append(np.nan)
            continue

        corrs.append(spearmanr(comparison, freqs).statistic)

    arr = np.array(corrs)
    return np.nanmean(arr), arr

# HELPERS & MAIN

def load_pickle(f_path):
    with open(f_path, 'rb') as in_f:
        data = pickle.load(in_f)
    return data

def get_hu19_defn(hu19_data, sense):
    """ Given a sense, return the definition.
    """
    word = sense.split('_')[0]
    return hu19_data[word][sense]['definition']

def query_HU19_sense_fac_example(query_sense):
    """
    """
    if not os.path.exists(COHA_SENSE_NN_EX_DIR):
        os.makedirs(COHA_SENSE_NN_EX_DIR)

    sense_nn = load_pickle(os.path.join(HU19_DIR, 'sense_nn.pkl'))
    hu19_data = load_pickle(os.path.join(HU19_DIR, 'prob_fitting_10.data'))
    conc_ratings = pd.read_csv(os.path.join(KUPER13_DIR, 'concreteness_ratings.csv'))[['Word', 'Conc.M']].set_index('Word').to_dict()['Conc.M']
    val_ratings = pd.read_csv(os.path.join(KUPER13_DIR, 'valence_ratings.csv'))[['Word', 'V.Mean.Sum']].set_index('Word').to_dict()['V.Mean.Sum']

    conc_val, conc_nns = eval_concreteness(query_sense, sense_nn, conc_ratings)
    val_val, val_nns = eval_valence(query_sense, sense_nn, val_ratings)
    demand_val, demand_nns = eval_demand(query_sense, sense_nn)
    
    records = []  # (query/nn, sense, defn, conc, val, demand)
    records += [('query', query_sense, get_hu19_defn(hu19_data, query_sense), conc_val, val_val, demand_val)]
    for i in tqdm(range(10)):
        records += [(
            f'nn_{i}',
            sense_nn[query_sense][i],
            get_hu19_defn(hu19_data, sense_nn[query_sense][i]),
            conc_nns[i],
            val_nns[i],
            demand_nns[i]
        )]

    df = pd.DataFrame.from_records(records, columns=['label', 'sense', 'definition', 'conc', 'val', 'demand'])
    df.to_csv(os.path.join(COHA_SENSE_NN_EX_DIR, f'{query_sense}.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-S', type=str, help='sense')

    args = parser.parse_args()

    query_HU19_sense_fac_example(args.S)
