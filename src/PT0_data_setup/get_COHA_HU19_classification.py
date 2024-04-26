import os
import sys
import re
import csv
import json
import shutil
import pickle
import argparse
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance

COHA_EMBEDS_DIR = '/ais/hal9000/datasets/COHA/embeds/hidden{}-masked{}'
# './local9000/COHA/embeds/hidden{}-masked{}'
COHA_EMBEDS_SENSE_DIR = '/ais/hal9000/datasets/COHA/embeds-hu19sense/hidden{}-masked{}'
# './local9000/COHA/embeds-hu19sense/hidden{}-masked{}'
HU19_DIR = './data/hu2019'

# https://ucrel.lancs.ac.uk/claws7tags.html
HU19_POS = {
    'noun': ['nd1', 'nn', 'nn1', 'nn2', 'nna', 'nnb', 'nnl1', 'nnl2', 'nno', 'nno2', 'nnt1', 'nnt2', 'nnu', 'nnu1', 'nnu2'], 
    'verb': ['vv0', 'vvd', 'vvg', 'vvgk', 'vvi', 'vvn', 'vvnk', 'vvz'], 
    'adjective': ['jj', 'jjr', 'jjt', 'jk'], 
    'adverb': ['ra', 'rex', 'rg', 'rgq', 'rgqv', 'rgr', 'rgt', 'rl', 'rp', 'rpk', 'rr', 'rrqv', 'rrr', 'rrt', 'rt'], 
    'preposition': ['if', 'ii', 'io', 'iw'], 
    'pronoun': ['pn', 'pn1', 'pnx1', 'pph1', 'ppho1', 'ppho2', 'pphs1', 'pphs2', 'ppio1', 'ppio2', 'ppis1', 'ppis2', 'ppx1', 'ppx2', 'ppy'],
    'exclamation': [], 
    'conjunction': ['cc', 'ccb', 'cs', 'csa', 'csn', 'cst', 'csw'], 
    'modal verb': ['vm', 'vmk'], 
    'determiner': ['at', 'at1', 'da', 'da1', 'da2', 'dar', 'dat', 'dd', 'dd1', 'dd2', 'ddq', 'ddqge', 'ddqv'],
    'plural noun': ['nn2', 'nnl2', 'nno2', 'nnt2', 'nnu2'], 
    'possessive determiner': ['appge'], 
    'auxiliary verb': ['vb0', 'vbdr', 'vbdz', 'vbg', 'vbi', 'vbm', 'vbn', 'vbr', 'vbz', 'vd0', 'vdd', 'vdg', 'vdi', 'vdn', 'vdz', 'vh0', 'vhd', 'vhg', 'vhi', 'vhn', 'vhz'],
    'abbreviation': [], 
    'cardinal number': ['mc', 'mc1', 'mc2', 'mcge'], 
    'interrogative adverb': ['rrq'], 
    'relative adverb': ['rrq'], 
    'possessive pronoun': ['appge', 'ppge'], 
    'proper noun': ['np', 'np1', 'np2', 'npd1', 'npd2', 'npm2'], 
    'infinitive marker': ['to'], 
    'interrogative pronoun': ['pnqo', 'pnqs', 'pnqv'], 
    'predeterminer': ['db', 'db2'],
    'relative pronoun': []
}


import io
class CPU_Unpickler(pickle.Unpickler):
    """ https://stackoverflow.com/a/71385473 
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def setup_dirs(target_dir):
    """ Remove existing data and create empty folders
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def filter_hu19_embeds(hu19_embeds, word):
    """ Return a dictionary containing only the sense affiliated with 'word'.
    """
    new_embeds = {}
    for sense in hu19_embeds:
        sense_word = sense.split('_')[0]
        if word == sense_word:
            new_embeds[sense] = hu19_embeds[sense]

    return new_embeds

def get_nearest_sense(word_embed, coha_pos, hu19_embeds_word, pos_ref):
    """ Return the nearest Hu et al. (2019) sense by name, for the inputted
    word embedding, and return the distance to the sense. 

    Heuristic: Seek to match the embedding and sense by POS, if possible.
    """
    # {'vvd_vvn@', 'vvd_jj@', 'vvn@_jj@', 'vvn@', 'vvz%_nn2', 'vv0%_nn1', 'nn1@', 'nn2', 'vvd_<sub>', 'vvg_nn1@', 'nn1_vv0%', 'vvd', 'nn1', 'vvd_jj@_vvn@', 'np1_<sub>', 'np1', 'vvg', 'vvn', 'nn1_<sub>', 'vv0%', 'nn1@_vvg'}

    # Obtain the cleaned POS tags
    coha_pos_tags = []
    for pos_tag in coha_pos.split('_'):
        if pos_tag != '<sub>':
            coha_pos_tags.append(pos_tag.strip('%@'))
    
    # Obtain sense and their POS tags
    hu19_senses = list(hu19_embeds_word.keys())
    hu19_senses_pos = [sense.split('_')[2] for sense in hu19_senses]  # 'noun', 'verb', etc

    matchable_senses = [pos_ref[pos_tag] for pos_tag in coha_pos_tags if ((pos_tag in pos_ref) and (pos_ref[pos_tag] in hu19_senses_pos))]  # 'vvd' --> 'verb'

    # Find the matching sense and distance
    nearest_sense, dist = '', np.inf 

    if matchable_senses:  # There are matchable POS tags
        for hu19_sense, hu19_sense_pos in zip(hu19_senses, hu19_senses_pos):
            if hu19_sense_pos in matchable_senses:
                cur_dist = distance.cosine(word_embed, hu19_embeds_word[hu19_sense])
                if cur_dist < dist:
                    nearest_sense = hu19_sense
                    dist = cur_dist
    else:  # General case, no clear matchable POS tags
        for hu19_sense in hu19_senses:
            cur_dist = distance.cosine(word_embed, hu19_embeds_word[hu19_sense])
            if cur_dist < dist:
                nearest_sense = hu19_sense
                dist = cur_dist

    return nearest_sense, dist

def classify_single_word(args):
    """
    """
    coha_embeds_f_name, hu19_embeds, coha_embeds_dir, coha_embeds_sense_dir, coha_pos_to_hu19_pos = args

    ### PREPARATION  
    word = os.path.splitext(coha_embeds_f_name)[0]

    # if word in ['but', 'at', 'not', 'do', 'on', 'you', 'as', 'with', 'for', 'his', 'it', 'he', 'that', 'have', 'in', 'to', 'and', 'of', 'be', 'the']:
    #     return

    coha_embeds_sense_path = os.path.join(coha_embeds_sense_dir, f'{word}.pkl')
    if os.path.exists(coha_embeds_sense_path):
        # already computed
        return

    coha_embeds_path = os.path.join(coha_embeds_dir, f'{word}.pkl')
    if not os.path.exists(coha_embeds_path):
        print(f'Sorry, no COHA embeddings file for {word} found')
        return

    try:
        with open(coha_embeds_path, 'rb') as in_f:
            word_entries = pickle.load(in_f)  # [{word, lemma, pos, year, sent_toks, sent_lems, sent_poss, embedding}, ...]
            # word_entries = CPU_Unpickler(in_f).load()
    except:
        print(f'Sorry, unable to open COHA embeddings file for {word}')
        return

    hu19_embeds_word = filter_hu19_embeds(hu19_embeds, word)
    if not hu19_embeds_word:
        print(f'Sorry, no HU19 embeddings available for {word} found')
        return

    ### CLASSIFICATION
    for word_entry in word_entries:
        word_embed = word_entry['embedding']
        nearest_sense, dist = get_nearest_sense(word_embed, word_entry['pos'], hu19_embeds_word, coha_pos_to_hu19_pos)
        word_entry['hu19_sense'] = nearest_sense
        word_entry['hu19_dist'] = dist

    ### SAVING
    with open(coha_embeds_sense_path, 'wb') as out_f:
        pickle.dump(word_entries, out_f)

def get_hu19_classification_COHA(H, M):
    """ Given embeddings files (as per COHA_EMBEDS_DIR), and the embeddings for
    a given sense, compute 
    """
    pool = mp.Pool(8)

    coha_embeds_dir = COHA_EMBEDS_DIR.format(H, M)
    coha_embeds_sense_dir = COHA_EMBEDS_SENSE_DIR.format(H, M)

    # setup_dirs(coha_embeds_sense_dir)

    hu19_embeds_path = os.path.join(HU19_DIR, 'diachronic_sense_emb.pkl')
    with open(hu19_embeds_path, 'rb') as in_f:
        hu19_embeds = pickle.load(in_f)  # sense --> embedding
    
    coha_pos_to_hu19_pos = {v: k for k, l in HU19_POS.items() for v in l}  # https://stackoverflow.com/a/71623853
    
    coha_embeds_f_names = os.listdir(coha_embeds_dir)
    # coha_embeds_f_names = [f'{word}.pkl' for word in ['but', 'at', 'not', 'do', 'on', 'you', 'as', 'with', 'for', 'his', 'it', 'he', 'that', 'have', 'in', 'to', 'and', 'of', 'be', 'the']]
    inputs = [(coha_embeds_f_name, hu19_embeds, coha_embeds_dir, coha_embeds_sense_dir, coha_pos_to_hu19_pos) 
              for coha_embeds_f_name in coha_embeds_f_names]
    # for inp in tqdm(inputs):
        # classify_single_word(inp)

    results = list(tqdm(pool.imap(classify_single_word, inputs), total=len(inputs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', type=int, default=1, help='no. hidden layers')
    parser.add_argument('-M', type=str, default='N', help='masking (Y or N)')

    args = parser.parse_args()

    get_hu19_classification_COHA(args.H, args.M)
