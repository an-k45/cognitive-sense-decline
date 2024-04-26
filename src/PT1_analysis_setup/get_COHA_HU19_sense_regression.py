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
from scipy.optimize import curve_fit
from scipy.linalg import lstsq

COHA_SENSE_PROPS_DIR = './data/COHA/hidden{}-masked{}/hu19_props'
COHA_SUMMARY_STATS_DIR = './results/COHA/stats/summary_56'
COHA_SENSE_REGRESSION_DIR = './data/COHA/hidden{}-masked{}/sense_regression'


def setup_dirs(target_dir):
    """ Remove existing data and create empty folders
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def get_hu19_senses_count(summary_path):
    """ Return a dictionary mapping word --> no. hu19 senses
    """
    df = pd.read_csv(summary_path)
    return df.set_index('word')['num_hu19_senses'].to_dict()

def decline_metric(props, decline_factor=5, min_decade=10, num_decades=20):
    """ 
    Declining: x(t) = a(b - t) if t <= b, 0 if t > b
        where: b in (0, 20)  # no. decades

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html
    https://stackoverflow.com/a/29384899

    The code is adapted from Francis et al. (2021): 
    https://github.com/ellarabi/linguistic_decline/blob/master/declining_list.py#L135C15-L135C15
    """
    if props[0] < 0.000001:  # 0.000005 in F21, /5(?) avg senses
        return 100

    total = sum(props)
    freq = [p / total for p in props]

    best_fit = 100
    best_dec = -1
    best_curve = []

    if max(freq) < decline_factor * freq[num_decades - 1]:
        return 100

    for zero_dec in range(2, num_decades):
        line = [zero_dec - i - 1 for i in range(0, zero_dec)]

        arr_freq = np.array(freq[0:zero_dec])
        arr_line = np.array(line)

        lin, a, b, c = lstsq(arr_line[:, np.newaxis], arr_freq)
        curve = [max((zero_dec - i - 1) * lin[0], 0) for i in range(0, num_decades)]

        arr_fit = np.array(curve)
        fit = np.linalg.norm(arr_fit - np.array(freq))

        if fit < best_fit:
            best_curve = curve
            best_fit = fit
            best_dec = zero_dec

    if best_dec < min_decade:  
        return 100

    return best_fit

def stable_metric(props, stable_factor=2.5):
    """ 
    The code is adapted from Francis et al. (2021): 
    https://github.com/ellarabi/linguistic_decline/blob/master/stable_list.py
    """
    if min(props) > 0 and max(props) / min(props) < stable_factor:
        total = sum(props)
        freq = [p / total for p in props]

        fit = np.linalg.norm(np.median(freq) - np.array(freq))

        return fit 

    return 100

def get_COHA_HU19_sense_regression(D, S):
    """ Produce two sets of senses: declining and stable.
    """
    coha_sense_props_dir = COHA_SENSE_PROPS_DIR.format(1, 'N')
    coha_summary_path = os.path.join(COHA_SUMMARY_STATS_DIR, 'coha_hu19_words_summary.csv')
    coha_sense_regression_dir = COHA_SENSE_REGRESSION_DIR.format(1, 'N')

    if not os.path.exists(coha_sense_regression_dir):
        os.makedirs(coha_sense_regression_dir)

    decline_records, stable_records = [], [] 
    word_hu19_senses_count = get_hu19_senses_count(coha_summary_path)

    for f_name in tqdm(os.listdir(coha_sense_props_dir)):
        df = pd.read_csv(os.path.join(coha_sense_props_dir, f_name))
        df_sorted = df.sort_values(by=['hu19_sense', 'year'])
        sense_props = df_sorted[['hu19_sense', 'prop_lexicon']].groupby('hu19_sense')['prop_lexicon'].apply(list).to_dict()  # https://stackoverflow.com/a/50505848

        for sense in sense_props:
            cur_props = sense_props[sense]
            cur_word = sense.split('_')[0]

            if cur_word not in word_hu19_senses_count:
                continue

            if len(cur_props) < 20:
                continue
                
            if cur_props[0] < 0.000001:  # 0.000005 in F21, /5(?) avg senses
                continue

            # stable_factor << decline_factor, s.t. no sense is both declining and stable
            decline_fit = decline_metric(cur_props, decline_factor=D)  # 100 is False
            stable_fit = stable_metric(cur_props, stable_factor=S)  # 100 is False

            sense_entry = {
                'word': cur_word,
                'hu19_sense': sense,
                'num_hu19_senses': word_hu19_senses_count[cur_word],
                'pos': sense.split('_')[2],
                'init_freq': sense_props[sense][0],
            }

            if decline_fit < 100:
                sense_entry['decline_metric'] = decline_fit
                decline_records.append(sense_entry)
            elif stable_fit < 100:
                sense_entry['stable_metric'] = stable_fit
                stable_records.append(sense_entry)
    
    df_decline = pd.DataFrame.from_records(decline_records).sort_values(by='hu19_sense')
    df_stable = pd.DataFrame.from_records(stable_records).sort_values(by='hu19_sense')

    print(f'declining senses: {len(df_decline.index)}')
    print(f'declining words:', df_decline['word'].nunique())
    print(f'stable senses: {len(df_stable.index)}')
    print(f'stable words:', df_stable['word'].nunique())

    df_decline.to_csv(os.path.join(coha_sense_regression_dir, f'decline-{D}.csv'), index=False)
    df_stable.to_csv(os.path.join(coha_sense_regression_dir, f'stable-{S}.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', type=float, default=5, help='decline factor')
    parser.add_argument('-S', type=float, default=2.5, help='stable factor')

    args = parser.parse_args()

    get_COHA_HU19_sense_regression(args.D, args.S)
