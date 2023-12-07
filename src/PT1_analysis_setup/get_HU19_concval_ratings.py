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
from sklearn import linear_model
from scipy.stats import pearsonr

COHA_SENSE_PROPS_DIR = './data/COHA/hidden1-maskedN/hu19_props'
COHA_SUMMARY56_DIR = './results/COHA/stats/summary_56'
HU19_DIR = './data/hu2019'
KUPER13_DIR = './data/kuperman13'


def load_pickle(f_path):
    with open(f_path, 'rb') as in_f:
        data = pickle.load(in_f)
    return data

def get_init_sense_props():
    """ Return a dictionary mapping each sense label to its initial frequency
    mapping. 
    """
    init_sense_props = {}

    for f_name in tqdm(os.listdir(COHA_SENSE_PROPS_DIR)):
        df = pd.read_csv(os.path.join(COHA_SENSE_PROPS_DIR, f_name))
        df_sorted = df.sort_values(by=['hu19_sense', 'year'])
        sense_props = df_sorted[['hu19_sense', 'prop_lexicon']].groupby('hu19_sense')['prop_lexicon'].apply(list).to_dict()  # https://stackoverflow.com/a/50505848

        for sense in sense_props:
            init_sense_props[sense] = sense_props[sense][0]
    
    return init_sense_props

def load_data(sense_embeds):
    """ Return a single dataframe, with word, sense, conc, val, and the full 
    768-dim embedding
    """
    df_embeds = pd.DataFrame.from_dict(sense_embeds, orient='index')

    df_conc = pd.read_csv(os.path.join(KUPER13_DIR, 'concreteness_ratings.csv'))[['Word', 'Conc.M']].rename(columns={"Word": "word", "Conc.M": "conc"})
    df_val = pd.read_csv(os.path.join(KUPER13_DIR, 'valence_ratings.csv'))[['Word', 'V.Mean.Sum']].rename(columns={"Word": "word", "V.Mean.Sum": "val"})

    df_embeds = df_embeds.reset_index().rename(columns={'index': 'sense'})
    df_embeds['word'] = df_embeds['sense'].apply(lambda x: x.split('_')[0])

    df = pd.merge(df_embeds, df_conc, on='word', how='left')
    df = pd.merge(df, df_val, on='word', how='left')

    return df

def load_sense_init_props(sense_embeds):
    """ Return the list of senses who occur LESS THAN 1/1M at init time.
    """
    init_sense_props = get_init_sense_props()

    senses = []

    for sense in list(sense_embeds.keys()):
        if sense not in init_sense_props:
            senses.append(sense) 
        elif init_sense_props[sense] < 0.000001:
            senses.append(sense)

    return senses

def load_sense_match_list():
    """
    """
    df_matches = pd.read_csv(os.path.join(HU19_DIR, 'matches', 'hu19_sense_matches_0.csv'))
    return df_matches['dec_sense'].to_list() + df_matches['stb_sense'].to_list()

def run_linear_regression(df, feature, removal):
    """ Fit and predict a linear regression model with x = the sense embedding
    and y = the feature rating (if it exists). Remove certain cases as passed in.

    Return the list of predicted values, and real values. 

    Helpful: 
    https://www.kaggle.com/code/shashankasubrahmanya/missing-data-imputation-using-regression
    https://towardsdatascience.com/a-guide-to-the-regression-of-rates-and-proportions-bcfe1c35344f
    """
    df_temp = df[~df['sense'].isin(removal)]
    df_notnan = df_temp[df_temp[feature].notnull()] # .reset_index(drop=True)
    missing = len(df_temp.index) - len(df_notnan.index)

    test_idxs = np.random.choice(df_notnan.index.to_numpy(), 1000, replace=False)
    df_notnan = df_notnan.drop(labels=test_idxs, axis=0)

    parameters = list(range(768))  # n dim of embeddings

    model = linear_model.LinearRegression()
    model.fit(X=df_notnan[parameters], y=df_notnan[feature])
    
    preds = model.predict(df[parameters])
    reals = np.array(df[feature].to_list())
    senses = df['sense'].to_list()

    preds_fix = np.interp(preds, (preds.min(), preds.max()), (np.nanmin(reals), np.nanmax(reals)))

    return preds_fix, reals, senses, test_idxs, len(df_notnan.index), missing

def get_correlation(preds, reals):
    good = ~np.logical_or(np.isnan(preds), np.isnan(reals))

    preds = preds[good]
    reals = reals[good]

    return round(pearsonr(preds, reals)[0], 4)

def get_HU19_concval_ratings():
    """
    Run the linear regression prediction model using...
    1. Only existing data
    2. (1), minus senses with less than 1/1M at init
    3. (1), minus senses in 0th match list
    4. (1), minus both conditions above

    Test the correlation between...
    1. Predicted data and real data
    2. (1), but only for senses in the 0th match list
    """
    if not os.path.exists(os.path.join(KUPER13_DIR, 'predictions')):
        os.makedirs(os.path.join(KUPER13_DIR, 'predictions'))

    sense_embeds = load_pickle(os.path.join(HU19_DIR, 'diachronic_sense_emb.pkl'))
    df = load_data(sense_embeds)
    
    init_senses = load_sense_init_props(sense_embeds)  # less than 1/1M
    match_senses = load_sense_match_list()  # matches set 0

    conds = {
        "all": [],
        "no_init": init_senses,
        "no_match": match_senses,
        "no_both": list(set(match_senses + init_senses))
    }

    records = []  # (conc/val, cond, missing, train_size, test_type, test_size, corr)

    for feature in ['conc', 'val']:
        # print('======================', feature)
        for cond_type, cond in tqdm(conds.items()):
            # print('=====', cond_type)
            
            preds, reals, senses, test_idxs, train_size, missing = run_linear_regression(df, feature, cond)
            # print(f'preds: {len(preds)}')
            # print(f'reals: {len(reals)}')

            df_ratings = pd.DataFrame.from_records(list(zip(senses, preds, reals)), columns=['sense', 'pred', feature])
            df_ratings.to_csv(os.path.join(KUPER13_DIR, 'predictions', f'{feature}_{cond_type}.csv'), index=False)

            corr = get_correlation(preds[test_idxs], reals[test_idxs])
            # print(corr)

            records += [(
                feature,
                cond_type,
                missing,
                train_size,
                'hold_out',
                1000,
                corr
            )]

            idxs = np.array([i for i in range(len(senses)) if senses[i] in match_senses])
            mask = np.ones(len(senses), dtype=bool)
            mask[idxs] = False

            preds = preds[~mask]
            reals = reals[~mask]
            # print(f'preds: {len(preds)}')
            # print(f'reals: {len(reals)}')

            corr = get_correlation(preds, reals)
            # print(corr)

            records += [(
                feature,
                cond_type,
                missing,
                train_size,
                'match_set',
                len(match_senses),
                corr
            )]

    df_records = pd.DataFrame.from_records(records, columns=['feature', 'cond', 'missing', 'train_size', 'test_type', 'test_size', 'corr'])
    df_records.to_csv(os.path.join(COHA_SUMMARY56_DIR, 'hu19_sense_concval_ratings.csv'), index=False)


if __name__ == "__main__":
    get_HU19_concval_ratings()
