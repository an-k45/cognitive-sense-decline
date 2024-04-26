import os
import sys
import re
import csv
import json
import random
import copy
import shutil
import pickle
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd

COHA_SENSE_REGRESSION_DIR = './data/COHA/hidden{}-masked{}/sense_regression'
HU19_DIR = './data/hu2019'
COHA_SUMMARY56_DIR = './results/COHA/stats/summary_56'


def setup_dirs(target_dir):
    """ Remove existing data and create empty folders
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def load_sense_10nn(path):
    """ Load the HU19 sense nearest neighbours, and trim to 10NN
    """
    with open(path, 'rb') as in_f:
        sense_nn = pickle.load(in_f)
    for sense in sense_nn:
        sense_nn[sense] = sense_nn[sense][:10]
    return sense_nn

def load_sense_regression_list(in_dir, f_name, threshold, sense_10nn):
    """ Load the given regression list (f_name). Keep the highest initial
    frequency sense of a word, remove any senses below the given threshold, and
    any senses with None POS. 

    Return it in dict form, including 10nn.
    """
    sense_list_path = os.path.join(in_dir, f_name)
    df_sense_list = pd.read_csv(sense_list_path).sort_values(by=['hu19_sense', 'init_freq'])

    metric = f_name.split('-')[0]

    # Clean 
    df_sense_list = df_sense_list.drop_duplicates(subset='word', keep='first')
    df_sense_list = df_sense_list.loc[df_sense_list[f'{metric}_metric'] <= threshold]
    df_sense_list = df_sense_list.loc[df_sense_list['pos'] != 'None']

    # Add in 10NN
    sense_dict = df_sense_list.set_index('hu19_sense').to_dict(orient='index')
    for sense in sense_dict:
        sense_dict[sense]['10nn'] = sense_10nn[sense]

    return sense_dict

def obtain_potential_matches(decline_senses, stable_senses):
    """ Return the list of potential matches for each declining sense
    """
    potential_matches = {sense: [] for sense in decline_senses} 

    for cur_dec_sense in decline_senses:
        dec_entry = decline_senses[cur_dec_sense]

        for cur_stb_sense in stable_senses:
            stb_entry = stable_senses[cur_stb_sense]

            dec_freq = dec_entry['init_freq']
            if not (dec_freq * 0.9 <= stb_entry['init_freq'] <= dec_freq * 1.1):
                continue

            if dec_entry['word'] == stb_entry['word']:
                continue

            if dec_entry['pos'] != stb_entry['pos']:
                continue

            dec_word_len = len(dec_entry['word'])
            stb_word_len = len(stb_entry['word'])
            if not (-2 <= dec_word_len - stb_word_len <= 2):
                continue

            if cur_dec_sense in stb_entry['10nn']:
                continue

            if not (-2 <= dec_entry['num_hu19_senses'] - stb_entry['num_hu19_senses'] <= 2):
                continue

            potential_matches[cur_dec_sense].append(cur_stb_sense)
    
    for sense in list(potential_matches.keys()):
        if len(potential_matches[sense]) == 0:
            del potential_matches[sense]

    return potential_matches

def obtain_final_matches_max_matches(potential_matches, stable_senses):
    """ Whittle down potential matches iteratively on two bases, in order of priority
     - Match senses which the fewest potential matches first (if empty, delete)
     - Minimize the total length difference between stable and declining senses

    Return the final matching pairs. 
    """
    stb_status = {sense: False for sense in stable_senses}  # Track whether a stable sense has been matched 
    final_matches = {}

    excess_chars = 0  # total decline - total stable --> neg means bias to stable, pos means bias to decline

    while len(potential_matches) > 0:
        # print(f'Total potential matches left: {len(potential_matches)}')

        # Find the sense with the fewest potential matches
        min_potential, min_dec_sense = np.inf, ''
        for dec_sense in potential_matches:
            num_potential = len(potential_matches[dec_sense])
            if num_potential < min_potential:
                min_potential = num_potential
                min_dec_sense = dec_sense

        # If multiple available, select the match that brings excess_chars to 0
        min_diff, min_stb_sense = np.inf, ''
        for stb_sense in potential_matches[min_dec_sense]:
            dec_word, stb_word = min_dec_sense.split('_')[0], stb_sense.split('_')[0]
            cur_diff = excess_chars + (len(dec_word) - len(stb_word))
            if abs(cur_diff) < abs(min_diff):
                min_diff = cur_diff
                min_stb_sense = stb_sense

        # Record the matching, remove it from potential, set stb_status to True, adjust char count
        final_matches[min_dec_sense] = min_stb_sense
        del potential_matches[min_dec_sense]
        stb_status[min_stb_sense] = True
        excess_chars += (len(min_dec_sense.split('_')[0]) - len(min_stb_sense.split('_')[0]))

        # Clear all potential matches of the newly matched sense, and clear any senses with 0 potential
        for dec_sense in list(potential_matches.keys()):
            potential_matches[dec_sense] = [stb_sense for stb_sense in potential_matches[dec_sense] if not stb_status[stb_sense]]
            if len(potential_matches[dec_sense]) == 0:
                del potential_matches[dec_sense]

    return final_matches, excess_chars

def obtain_final_matches_min_chars(potential_matches, stable_senses, pick_random):
    """ Whittle down potential matches iteratively on two bases, in order of priority
     - Minimize the total length difference between stable and declining senses
     - Match senses which the fewest potential matches first (if empty, delete)

    Return the final matching pairs. 
    """
    stb_status = {sense: False for sense in stable_senses}  # Track whether a stable sense has been matched 
    final_matches = {}

    excess_chars = 0  # total decline - total stable --> neg means bias to stable, pos means bias to decline

    while len(potential_matches) > 0:
        # print(f'Total potential matches left: {len(potential_matches)}')

        # Find the match(es) which keeps excess_chars closest to 0, among every sense
        min_diff = np.inf 
        for dec_sense in potential_matches:
            for stb_sense in potential_matches[dec_sense]:
                dec_word, stb_word = dec_sense.split('_')[0], stb_sense.split('_')[0]
                cur_diff = excess_chars + (len(dec_word) - len(stb_word))
                if abs(cur_diff) < abs(min_diff):
                    min_diff = cur_diff
        
        min_matches = []  # [(min_diff, min_dec_sense, min_stb_sense, num_potential_matches)]
        for dec_sense in potential_matches:
            for stb_sense in potential_matches[dec_sense]:
                dec_word, stb_word = dec_sense.split('_')[0], stb_sense.split('_')[0]
                cur_diff = excess_chars + (len(dec_word) - len(stb_word))
                if abs(min_diff) == abs(cur_diff):
                    min_matches.append((
                        cur_diff,
                        dec_sense,
                        stb_sense,
                        len(potential_matches[dec_sense])
                    ))

        # If multiple available, pick the one with the fewest potential matches
        min_matches.sort(key=lambda x: x[3])
        min_matches = [min_match for min_match in min_matches if min_match[3] == min_matches[0][3]]
        if pick_random:
            random.shuffle(min_matches)
        min_dec_sense = min_matches[0][1]
        min_stb_sense = min_matches[0][2]

        # Record the matching, remove it from potential, set stb_status to True, adjust char count
        final_matches[min_dec_sense] = min_stb_sense
        del potential_matches[min_dec_sense]
        stb_status[min_stb_sense] = True
        excess_chars += (len(min_dec_sense.split('_')[0]) - len(min_stb_sense.split('_')[0]))

        # Clear all potential matches of the newly matched sense, and clear any senses with 0 potential
        for dec_sense in list(potential_matches.keys()):
            potential_matches[dec_sense] = [stb_sense for stb_sense in potential_matches[dec_sense] if not stb_status[stb_sense]]
            if len(potential_matches[dec_sense]) == 0:
                del potential_matches[dec_sense]

    return final_matches, excess_chars

def cut_excess_char_matches(matches, excess_chars):
    """ Given the a list of matches, and the excess chars (+ bias to decline, 
    - bias to stable), lob off matches until we are within a difference of 1. 
    """
    cur_char_diff = excess_chars
    cur_matches = list(matches.items())
    num_matches = len(cur_matches)

    for i in range(num_matches - 1, 0 - 1, -1):
        dec_sense, stb_sense = cur_matches[i][0], cur_matches[i][1]
        sense_diff = len(dec_sense) - len(stb_sense)
        pred_char_diff = cur_char_diff + sense_diff

        if abs(pred_char_diff) < abs(cur_char_diff) and abs(cur_char_diff) > 1:
            cur_matches.pop(i)
            cur_char_diff = pred_char_diff
    
    if len(cur_matches) % 2 == 1:
        cur_char_diff = cur_char_diff + (len(cur_matches[0][0]) - len(cur_matches[0][1]))
        cur_matches.pop(0)

    return cur_matches, cur_char_diff

def get_HU19_sense_matchings():
    """ Produce a pair of matches between declining and stable senses where senses:
     - Initial frequencies are within 10% of each other
     - Belong to different words
     - Match in POS
     - Share a length of -+ 2
     - Not be within the 10 nearest neighbours of another
     - Belong to words within 2 total senses of each other

    Additionally
     - For words with multiple declining or stable senses, use the one with the
       highest initial frequency only 
     - Minimize the total length difference between stable and declining senses
       which are matched
    """
    sense_nn_path = os.path.join(HU19_DIR, 'sense_nn.pkl')
    coha_sense_regression_dir = COHA_SENSE_REGRESSION_DIR.format(1, 'N')
    sense_matches_dir = os.path.join(HU19_DIR, 'matches')
    if not os.path.exists(sense_matches_dir):
        os.makedirs(sense_matches_dir)

    print('Load 10NN of each sense as a dict')
    sense_10nn = load_sense_10nn(sense_nn_path)
    
    print('Load two sense regression lists')
    # Keys: 'word', 'num_hu19_senses', 'pos', 'init_freq', 'decline_metric', '10nn'
    decline_senses = load_sense_regression_list(coha_sense_regression_dir, 'decline-5.0.csv', 0.15, sense_10nn)
    stable_senses = load_sense_regression_list(coha_sense_regression_dir, 'stable-2.5.csv', 0.055, sense_10nn)

    print('Obtain the potential matches for each declining sense')
    potential_matches = obtain_potential_matches(decline_senses, stable_senses)

    summary_info = []
    for i in tqdm(range(10)):
        pick_random = True if i > 0 else False
        cur_potential_matches = copy.deepcopy(potential_matches)

        # print('Whittle down potential matches')
        # final_matches, excess_chars = obtain_final_matches_max_matches(potential_matches, stable_senses)
        final_matches, excess_chars = obtain_final_matches_min_chars(cur_potential_matches, stable_senses, pick_random)

        final_matches, excess_chars = cut_excess_char_matches(final_matches, excess_chars)

        summary_info.append((len(decline_senses), len(stable_senses), len(final_matches), excess_chars))

        hu19_matches_path = os.path.join(sense_matches_dir, f'hu19_sense_matches_{i}.csv')
        # records = list(final_matches.items())
        df = pd.DataFrame.from_records(final_matches, columns=['dec_sense', 'stb_sense'])
        df.to_csv(hu19_matches_path, index=False)

    df_summary = pd.DataFrame.from_records(summary_info, columns=['num_dec_senses', 'num_stb_senses', 'num_matches', 'excess_chars'])
    df_summary.to_csv(os.path.join(COHA_SUMMARY56_DIR, 'hu19_sense_matches_summary.csv'), index=False)


if __name__ == "__main__":
    get_HU19_sense_matchings()
