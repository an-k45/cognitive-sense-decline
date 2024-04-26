import os
import sys
import re
import csv
import json
import shutil
import pickle
import argparse
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance

COHA_EMBEDS_SENSE_DIR = '/ais/hal9000/datasets/COHA/embeds-hu19sense/hidden{}-masked{}'
# './local9000/COHA/embeds-hu19sense/hidden{}-masked{}'
COHA_SENSE_PROPS_DIR = './data/COHA/hidden{}-masked{}/hu19_props'
COHA_SUMMARY56_DIR = './results/COHA/stats/summary_56'
COHA_SUMMARY1234_DIR = './results/COHA/stats/summary_1234'

CSV_HEADERS = ['word', 'sense', 'pos', 'year', 'num_sents', 'prop_word', 'prop_lexicon']
SUB_WORDS = ['but', 'at', 'not', 'do', 'on', 'you', 'as', 'with', 'for', 'his', 'it', 'he', 'that', 'have', 'in', 'to', 'and', 'of', 'be', 'the']


def setup_dirs(target_dir):
    """ Remove existing data and create empty folders
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def round_down(m, n):
    """ https://stackoverflow.com/a/68659666
    """
    return m // n * n

def get_sense_props(word_entries, df_coha_temp_counts, ratio):
    """ Return a DataFrame adhering to CSV_HEADERS
    """
    # [word, pos, year, hu19_sense]
    df = pd.DataFrame.from_records(word_entries, exclude=['lemma', 'sent_toks', 'sent_lems', 'sent_poss', 'embedding', 'hu19_dist'])
    df['year'] = df['year'].apply(lambda x: round_down(x, 10))
    df_counts = df[['year', 'hu19_sense']].value_counts().reset_index(name='num_sents').sort_values(['year', 'hu19_sense'])
    df_counts['pos'] = df_counts['hu19_sense'].apply(lambda x: x.split('_')[2])

    df_per_decade = df_counts.groupby('year').sum('num_sents').reset_index()

    df_counts['prop_word'] = df_counts['num_sents'].div(df_counts['year'].map(df_per_decade.set_index('year')['num_sents']))
    df_counts['prop_lexicon'] = df_counts['num_sents'].div(df_counts['year'].map(df_coha_temp_counts.set_index('year')['num_sents']))  # https://stackoverflow.com/a/64723941
    df_counts['prop_lexicon'] *= ratio  # Adjust the 20 subsampled words to (approx) their true proportions

    return df_counts

def get_COHA_HU19_sense_props(H, M):
    """ Compute the per-word and per-lexicon proportions of senses, per decade,
    and save them to a CSV per word, of the form CSV_HEADERS. 
    """
    coha_embeds_sense_dir = COHA_EMBEDS_SENSE_DIR.format(H, M)
    coha_sense_props_dir = COHA_SENSE_PROPS_DIR.format(H, M)
    # setup_dirs(coha_sense_props_dir)

    coha_temporal_counts_path = os.path.join(COHA_SUMMARY56_DIR, 'coha_temporal_counts.csv')
    df_coha_temp_counts = pd.read_csv(coha_temporal_counts_path)

    coha_hu19_summary_path = os.path.join(COHA_SUMMARY1234_DIR, f'hu19_words_summary_0.0.csv')
    df_coha_hu19_summary = pd.read_csv(coha_hu19_summary_path)
    total_num_sents = df_coha_hu19_summary[['word', 'num_sents']].set_index('word').to_dict()['num_sents']

    for f_word in tqdm(os.listdir(coha_embeds_sense_dir)):
        word = os.path.splitext(f_word)[0]
        
        coha_sense_props_path = os.path.join(coha_sense_props_dir, f'{word}.csv')

        if os.path.exists(coha_sense_props_path):
            continue

        coha_embeds_sense_path = os.path.join(coha_embeds_sense_dir, f'{word}.pkl')
        if not os.path.exists(coha_embeds_sense_path):
            print(f'Sorry, no COHA embeddings file for {word} found')
            continue

        with open(coha_embeds_sense_path, 'rb') as in_f:
            # [{word, lemma, pos, year, sent_toks, sent_lems, sent_poss, embedding, hu19_sense, hu19_dist}, ...]
            word_entries = pickle.load(in_f)  

        num_entries = len(word_entries)
        if num_entries == 0:
            continue
        ratio = 1 / (num_entries / total_num_sents[word])
        if word not in SUB_WORDS:
            ratio = 1

        try:
            df_sense_props = get_sense_props(word_entries, df_coha_temp_counts, ratio)
            df_sense_props.to_csv(coha_sense_props_path, index=False)
        except: 
            print(f'Failed on: {f_word}')
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', type=int, default=1, help='no. hidden layers')
    parser.add_argument('-M', type=str, default='N', help='masking (Y or N)')

    args = parser.parse_args()

    get_COHA_HU19_sense_props(args.H, args.M)
