import os
import re
import csv
import shutil
import multiprocessing as mp
import time

from tqdm import tqdm
tqdm.pandas()
import pandas as pd


COHA_TAGGED_DIR = '/ais/hal9000/datasets/COHA/clean/tagged'
# './local9000/COHA/clean/tagged'
COHA_WORDS_DIR = '/ais/hal9000/datasets/COHA/words'
# './local9000/COHA/words'

CSV_HEADERS = ['word', 'lemma', 'pos', 'year', 'sent_toks', 'sent_lems', 'sent_poss']
QUEUE_WAIT = 60


def setup_dirs(target_dir):
    """ Remove existing data and create empty folders
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def get_word_info(args):
    """
    """
    L, tagged_dir, tagged_txt = args

    year = tagged_txt.split("_")[1]
    in_path = os.path.join(COHA_TAGGED_DIR, tagged_dir, tagged_txt)

    with open(in_path, 'r') as in_f:
        cur_sent = []  # [(word, lemma, pos), ...]
        for line in in_f:
            try:
                word, lemma, pos = line.strip('\n').split('\t')

                if lemma == '<nul>':  
                    continue

                if word != '<eos>':
                    cur_sent.append((word, lemma, pos))
                    continue

                # We are at EOS
                toks = [item[0].lower() for item in cur_sent]
                lems = [item[1] for item in cur_sent]
                poss = [item[2] for item in cur_sent]

                sent_toks = " ".join(toks)
                sent_lems = " ".join(lems)
                sent_poss = " ".join(poss)
                
                for tok, lem, pos in zip(toks, lems, poss):
                    word_info = {
                        'word': tok, 
                        'lemma': lem, 
                        'pos': pos, 
                        'year': year, 
                        'sent_toks': sent_toks, 
                        'sent_lems': sent_lems, 
                        'sent_poss': sent_poss
                    }

                    if not word_info['lemma'].isalpha():
                        continue
                    
                    # if lem not in D:
                    #     D[lem] = []
                    # D[lem].append(word_info)

                    L.append(word_info)

                    # out_path = os.path.join(COHA_WORDS_DIR, f'{lem}.csv')
                    # write_csv_line(out_path, word_info)
            except:
                pass

            cur_sent = []

def write_word_info_by_lemma(args):
    """
    """
    D, lemma = args

    out_path_csv = os.path.join(COHA_WORDS_DIR, f'{lemma}.csv')
    file_exists = os.path.exists(out_path_csv)

    with open(out_path_csv, 'a') as in_f:
        writer = csv.DictWriter(in_f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        # for row in D[lemma]:
        #     writer.writerow(row)
        writer.writerows(D[lemma])

def sort_word_files(word_file):
    """ Sort word files by cols
    """
    in_path = os.path.join(COHA_WORDS_DIR, word_file)
    try:
        df = pd.read_csv(in_path)
        df_new = df.sort_values(by=['year', 'pos'])
        df_new.to_csv(in_path, index=False)
    except:
        os.remove(in_path)

def parse_COHA():
    """
    """
    setup_dirs(COHA_WORDS_DIR)
    manager = mp.Manager()
    pool = mp.Pool(mp.cpu_count() + 2)

    # Run each step iteratively on a decade
    tagged_dirs = [dir_name for dir_name in os.listdir(COHA_TAGGED_DIR) if ".zip" not in dir_name]

    print("Parsing (clean) COHA into word files...")
    for tagged_dir in tqdm(tagged_dirs):
        # 0. Set up
        L = [] # manager.list()  
        # D = manager.dict()  # {lemma: [word_info, ...]}

        # 1. Parse all files, and save them to a dict of {lemma: [word_info, ...]}
        tagged_txts = os.listdir(os.path.join(COHA_TAGGED_DIR, tagged_dir))
        # inputs1 = [(L, tagged_dir, tagged_txt) for tagged_txt in tagged_txts]
        # results1 = list(tqdm(pool.imap(get_word_info, inputs1), total=len(inputs1)))  # https://stackoverflow.com/a/45276885
        for tagged_txt in tqdm(tagged_txts):
            get_word_info((L, tagged_dir, tagged_txt))
        
        print("Converting to dataframe...")
        df1 = pd.DataFrame(L)
        print("Groupby...")
        df1 = df1.groupby('lemma')
        print("Apply...")
        df1 = df1.progress_apply(lambda x: x.to_dict(orient='r'))  # https://stackoverflow.com/a/54132728
        print("to dict...")
        D = df1.to_dict()

        # for item in tqdm(L):
        #     lem = item['lemma']
        #     if lem not in D:
        #         D[lem] = manager.list()
        #     D[lem].append(item)

        # 2. Iterate through lemma's, and write to file 
        lemmas = D.keys()
        # inputs2 = [(D, lemma) for lemma in lemmas]
        # results2 = list(tqdm(pool.imap(write_word_info_by_lemma, inputs2), total=len(inputs2)))
        for lemma in tqdm(lemmas):
            write_word_info_by_lemma((D, lemma))

    print("Sorting all word files by year and pos...")
    word_files = os.listdir(COHA_WORDS_DIR)
    results = list(tqdm(pool.imap(sort_word_files, word_files), total=len(word_files)))

    pool.close()
    pool.join()


if __name__ == "__main__":
    parse_COHA()