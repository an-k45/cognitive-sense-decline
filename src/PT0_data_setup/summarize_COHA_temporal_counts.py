import os
import sys
import re
import csv
import json
import shutil
import pickle
import argparse
import multiprocessing as mp
from collections import Counter

from tqdm import tqdm
import numpy as np
import pandas as pd

COHA_WORDS_DIR = '/ais/hal9000/datasets/COHA/words'
# './local9000/COHA/words'
COHA_SUMMARY_DIR = './results/COHA/stats/summary_56'
HU19_DIR = './data/hu2019'

CSV_HEADERS = ['year', 'num_words', 'num_sents', 'num_hu19_words', 'num_hu19_sents', 'prop_hu19_words', 'prop_hu19_sents']


def round_down(m, n):
    """ https://stackoverflow.com/a/68659666
    """
    return m // n * n

def get_word_temporal_count(args):
    """ Given a single word file name through multiprocessing, append a 
    dictionary which denotes the number of usages in each Y=10 length timespan.
    Additionally, save the word so that we can combine HU19 and all words 
    separately. 
    """
    L, coha_f_name = args

    in_path = os.path.join(COHA_WORDS_DIR, coha_f_name)
    df = pd.read_csv(in_path)  # word, lemma, pos, year, sent_toks, sent_lems, sent_poss

    per_year_counts = df['year'].value_counts().to_dict()
    
    time_periods = list(range(1810, 2009, 10))
    per_decade_counts = {period: 0 for period in time_periods}

    for year in per_year_counts.keys():
        decade = round_down(year, 10)
        per_decade_counts[decade] += per_year_counts[year]
    
    L.append((
        os.path.splitext(coha_f_name)[0],
        per_decade_counts
    ))

def collect_temporal_counts():
    """ Use multiprocessing to return a list containing tuples of the form:
     - word
     - dict of counts per decade
    """
    manager = mp.Manager()
    L = manager.list()
    pool = mp.Pool(8)

    coha_f_names = os.listdir(COHA_WORDS_DIR)
    inputs = [(L, coha_f_name) for coha_f_name in coha_f_names]
    print('Collected inputs...')
    results = list(tqdm(pool.imap(get_word_temporal_count, inputs), total=len(inputs)))

    pool.close()
    pool.join()

    L = list(L)
    return L

def summarize_COHA_temporal_counts():
    """ Save a CSV that summarizes the total number of sentence counts, among
    other statistics in CSV_HEADERS, for timespans of length Y=10, across the COHA 
    period. 
    """
    hu19_data_path = os.path.join(HU19_DIR, './prob_fitting_10.data')
    with open(hu19_data_path, 'rb') as in_f:
        hu19_data = pickle.load(in_f)
    hu19_words = list(hu19_data.keys())

    if not os.path.exists(COHA_SUMMARY_DIR):
        os.makedirs(COHA_SUMMARY_DIR)
    out_path = os.path.join(COHA_SUMMARY_DIR, 'coha_temporal_counts.csv')

    print('Collecting temporal counts...')
    temporal_counts = collect_temporal_counts()

    time_periods = list(range(1810, 2009, 10))
    full_temporal_count = Counter({period: 0 for period in time_periods})
    hu19_temporal_count = Counter({period: 0 for period in time_periods})
    full_occurrence_count = Counter({period: 0 for period in time_periods})
    hu19_occurrence_count = Counter({period: 0 for period in time_periods})

    print('Summing counts into decades...')
    for word, word_temporal_count in tqdm(temporal_counts):
        counts = Counter(word_temporal_count)
        occurrence = Counter({period: (0 if word_temporal_count[period] == 0 else 1) for period in word_temporal_count})

        full_temporal_count.update(counts)
        full_occurrence_count.update(occurrence)

        if word in hu19_words:
            hu19_temporal_count.update(counts)
            hu19_occurrence_count.update(occurrence)
    
    print('Creating records and saving...')
    records = []
    for period in time_periods:
        record = (
            period,
            full_occurrence_count[period],
            full_temporal_count[period],
            hu19_occurrence_count[period],
            hu19_temporal_count[period],
            round(hu19_occurrence_count[period] / full_occurrence_count[period], 4),
            round(hu19_temporal_count[period] / full_temporal_count[period], 4),
        )
        records.append(record)
    
    df = pd.DataFrame.from_records(records, columns=CSV_HEADERS)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    summarize_COHA_temporal_counts()