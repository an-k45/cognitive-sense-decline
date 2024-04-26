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

COHA_EMBEDS_DIR = '/ais/hal9000/datasets/COHA/embeds-hu19sense/hidden{}-masked{}'
# './local9000/COHA/embeds-hu19sense/hidden{}-masked{}'
COHA_SUMMARY_DIR = './results/COHA/stats/summary_56'
HU19_DIR = './data/hu2019'

CSV_HEADERS = ['word', 
               'num_sents', 'mean_sent_len', 'med_sent_len', 
               'mean_year', 'med_year', 
               'num_years', 'min_year', 'max_year',
               'num_hu19_senses', 'max_hu19_jsd']


def get_max_jsd(word_data):
    """ For a set of word data from Hu et al. (2019), return the maximum
    Jensen-Shannon Divergences from the initial year to each year. 
    """
    arr = []
    for sense in word_data:
        arr.append(word_data[sense]['y'])
    
    arr = np.asarray(arr)
    t0 = arr[:,0]

    jsds = []
    for i in range(np.shape(arr)[1]):
        ti = arr[:,i]
        jsd = distance.jensenshannon(t0, ti, 2)
        jsds.append(jsd)
    
    return round(max(jsds), 3)

def get_word_summary(word, hu19_data, in_path):
    """
    """
    with open(in_path, 'rb') as in_f:
        word_entries = pickle.load(in_f)  # word, lemma, pos, year, sent_toks, sent_lems, sent_poss, embedding
    df = pd.DataFrame.from_records(word_entries, exclude=['embedding'])  # word, lemma, pos, year, sent_toks, sent_lems, sent_poss

    num_sents = len(df.index)
    mean_sent_len = df['sent_toks'].str.split().apply(len).mean().round(2)
    med_sent_len = df['sent_toks'].str.split().apply(len).median().round(2)

    mean_year = df['year'].mean().round(1)
    med_year = df['year'].median().round(1)

    num_years = df['year'].nunique()
    min_year = df['year'].min()
    max_year = df['year'].max()

    word_summary = {
        'word': word, 
        'num_sents': num_sents, 
        'mean_sent_len': mean_sent_len, 
        'med_sent_len': med_sent_len,
        'mean_year': mean_year, 
        'med_year': med_year, 
        'num_years': num_years,
        'min_year': min_year, 
        'max_year': max_year,
        'num_hu19_senses': len(hu19_data[word]), 
        'max_hu19_jsd': get_max_jsd(hu19_data[word]),
    }

    return word_summary

def summarize_COHA_HU19_words():
    """ Provide summary statistics (outlined in CSV_HEADERS) of words in COHA.
    This script only provides summary stats for words in Hu et al. (2019). 
    """
    hu19_data_path = os.path.join(HU19_DIR, 'prob_fitting_10.data')
    with open(hu19_data_path, 'rb') as in_f:
        hu19_data = pickle.load(in_f)
    hu19_words = list(hu19_data.keys())

    coha_embeds_dir = COHA_EMBEDS_DIR.format(1, 'N')
    out_path = os.path.join(COHA_SUMMARY_DIR, 'coha_hu19_words_summary.csv')

    word_summaries = []
    for word in tqdm(hu19_words):
        try:
            coha_embeds_path = os.path.join(coha_embeds_dir, f'{word}.pkl')
            if os.path.exists(coha_embeds_path) and hu19_data[word]:
                word_summary = get_word_summary(word, hu19_data, coha_embeds_path)
                word_summaries.append(word_summary)
        except:
            continue

    df = pd.DataFrame(word_summaries)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    summarize_COHA_HU19_words()
