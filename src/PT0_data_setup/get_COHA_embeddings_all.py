import os
import sys
import re
import csv
import json
import shutil
import pickle
import argparse

from tqdm import tqdm
import pandas as pd
import torch
from transformers import BertModel, BertTokenizerFast  # BertTokenizer

COHA_WORDS_DIR = '/ais/hal9000/datasets/COHA/words/'
# './local9000/COHA/words/'
COHA_EMBEDS_DIR = '/ais/hal9000/datasets/COHA/embeds-temp/hidden{}-masked{}'
# './local9000/COHA/embeds-temp/hidden{}-masked{}'
COHA_SUMMARY_DIR = './results/COHA/stats/summary_1234'

SAMPLE_SIZE = 2000000


def setup_dirs(target_dir):
    """ Remove existing data and create empty folders
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def get_hu19_words(coha_hu19_summary_path):
    """ Return a list of words from Hu et al. (2019), in order of lowest to
    highest frequency. 
    """
    
    df_coha_hu19_summary = pd.read_csv(coha_hu19_summary_path)
    words = df_coha_hu19_summary['word'].to_list()
    words.reverse()

    return words

def get_pretok_sent(toks):
    """ Reconstruct the sentence from a set of tokens, using the ## values to 
    infer where splits occurred. 

    Adapted from https://stackoverflow.com/a/66238762
    """
    pretok_sent = ""
    for tok in toks:
        if tok in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        if tok.startswith("##"):
            pretok_sent += tok[2:]
        else:
            pretok_sent += " " + tok
    return pretok_sent[1:]

def get_index_of_word(sent_words, word, num_repeats):
    """ Given a sentence as a list, return the n-th (num_repeats) index of the 
    occurrence of the word, which may in fact be a morphological derivative of 
    it. In such a case, we will just check for subset. 
    """
    count = 0
    if word in sent_words:
        for i, cur_word in enumerate(sent_words):
            if cur_word == word:
                count += 1
                if count == num_repeats:
                    return i
    raise ValueError

def get_embeddings_COHA_all(H, M):
    """
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()

    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(1)
    model = model.to(device)

    coha_words_dir = COHA_WORDS_DIR.format(H, M)
    coha_embeds_dir = COHA_EMBEDS_DIR.format(H, M)
    setup_dirs(coha_embeds_dir)

    coha_hu19_summary_path = os.path.join(COHA_SUMMARY_DIR, f'hu19_words_summary_0.0.csv')
    # words = ['but', 'at', 'not', 'do', 'on', 'you', 'as', 'with', 'for', 'his', 'it', 'he', 'that', 'have', 'in', 'to', 'and', 'of', 'be', 'the']
    words = get_hu19_words(coha_hu19_summary_path)  # Lowest freq to highest freq of all words in HU19 that occur in COHA

    for cur_word_idx, cur_word in tqdm(enumerate(words), total=len(words)):
        if cur_word_idx % 100 == 0:
            print(f'Iteration: {cur_word_idx}, word: {cur_word}')

        try: 
            in_path = os.path.join(coha_words_dir, f'{cur_word}.csv')
            out_path = os.path.join(coha_embeds_dir, f'{cur_word}.pkl')

            df = pd.read_csv(in_path)
            if len(df.index) > SAMPLE_SIZE:  # Sometimes the word has millions of sents -- lots of computing time
                df = df.sample(n=SAMPLE_SIZE, random_state=1)

            items = df.values.tolist()  # ['word', 'lemma', 'pos', 'year', 'sent_toks', 'sent_lems', 'sent_poss']

            word_entries = []

            batch_size = 32
            iterations = len(items) // batch_size

            with torch.no_grad():  # This tells the model not to adjust its weights
                for i in range(iterations + 1): # tqdm(range(iterations + 1)):
                    try:
                        sub = items[i*batch_size:(i+1)*batch_size]
                        sub_sents = [item[4] for item in sub]
                        tokenized_sentences = tokenizer.batch_encode_plus(sub_sents, return_tensors='pt', add_special_tokens=True, padding=True, truncation=True)
                        tokenized_sentences = tokenized_sentences.to(device)

                        # Compute the embeddings for the batch, including OOV embeddings
                        output = model(**tokenized_sentences)
                        # wordpieces_encoded = output[0]  # has shape [n, k, 768] for [n_sentences, max_sentence_length, embedding_dim]
                        # sentence_encoded = output[1]    # has shape [n, 768] for [n_sentences, embedding_dim]
                        hidden_states = output[2]       # has shape [13, n, k, 768] for [h_layers, n_sentences, max_sentence_length, embedding_dim]

                        # has shape [n, k, 13, 768] for [n_sentences, max_sentence_length, h_layers, embedding_dim]
                        hidden_embeddings = torch.stack(hidden_states, dim=0).permute(1,2,0,3)

                        num_repeat_sents, prev_sent = 1, ""  # Repeats occur because 'word' occurs multiple times in a sentence

                        for j, sent_embed in enumerate(hidden_embeddings):
                            try:
                                if sub[j][4] == prev_sent:
                                    num_repeat_sents += 1
                                else: 
                                    num_repeat_sents = 1

                                cur_word = sub[j][0]
                                tokens = tokenized_sentences[j].tokens
                                sent_words = get_pretok_sent(tokens).split()
                                cur_word_idx = get_index_of_word(sent_words, cur_word, num_repeat_sents)
                                tok_to_word_idxs = tokenized_sentences[j].word_ids
                                token_idxs = [idx for idx, value in enumerate(tok_to_word_idxs) if value == cur_word_idx]

                                token_vecs = []
                                for token_idx in token_idxs:
                                    sum_vec = torch.sum(sent_embed[token_idx][-H:], dim=0)
                                    token_vecs.append(sum_vec)
                                word_vec = torch.mean(torch.stack(token_vecs, dim=0), dim=0)

                                # Save these as entries (of the form above) to a list
                                word_entry = {
                                    'word': cur_word, 
                                    'lemma': sub[j][1], 
                                    'pos': sub[j][2], 
                                    'year': sub[j][3], 
                                    'sent_toks': sub[j][4], 
                                    'sent_lems': sub[j][5], 
                                    'sent_poss': sub[j][6],
                                    'embedding': word_vec.cpu().numpy()
                                }
                                word_entries.append(word_entry)

                                prev_sent = sub[j][4]
                            except:
                                continue   
                    except:
                        continue
                    torch.cuda.empty_cache()
            
            # Remove duplicates
            df_word_entries = pd.DataFrame(word_entries)
            df_dups_free = df_word_entries.drop_duplicates(subset='sent_toks', keep='first')  # https://stackoverflow.com/a/13059751
            word_entries_clean = df_dups_free.to_dict('records')

            # Once we've iterated through the whole file, save the list to a pkl file
            with open(out_path, 'wb') as out_f:
                pickle.dump(word_entries_clean, out_f)
        except:
            print(f'Failed on: {cur_word}')
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', type=int, default=1, help='no. hidden layers')
    parser.add_argument('-M', type=str, default='N', help='masking (Y or N)')

    args = parser.parse_args()

    get_embeddings_COHA_all(args.H, args.M)
