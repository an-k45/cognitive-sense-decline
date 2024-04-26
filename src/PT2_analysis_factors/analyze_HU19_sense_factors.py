import os
import sys
import re
import csv
import json
import shutil
import pickle
import argparse
from random import Random
from collections import Counter

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import statsmodels.api as sm

COHA_SENSE_PROPS_DIR = './data/COHA/hidden1-maskedN/hu19_props'
COHA_SENSE_ANALYSIS_DIR = './data/COHA/hidden1-maskedN/sense_analysis_all10nn_matnanmean'
COHA_SENSE_ANALYSIS_SUMMARY_DIR = './data/COHA/hidden1-maskedN/sense_analysis_summary_all10nn_matnanmean'
COHA_SENSE_REGRESSION_DIR = './data/COHA/hidden1-maskedN/sense_regression'
COHA_CDIV_DIR = './data/COHA/hidden1-maskedN/cdiv/scores'
HU19_DIR = './data/hu2019'
KUPER13_DIR = './data/kuperman13'
COHA_SUMMARY56_DIR = './results/COHA/stats/summary_56'

### HELPERS

def load_pickle(f_path):
    with open(f_path, 'rb') as in_f:
        data = pickle.load(in_f)
    return data

def get_word_from_sense(sense):
    """ Given a sense of the form 'word_1_pos_1' return 'word'. 
    """
    return sense.split('_')[0]

def get_sense_init_df(cur_sense):
    """ Return a dataframe of sense information at time 1810 for senses that meet the 1/1M threshold
    """
    word = cur_sense.split('_')[0]
    df_word_props = pd.read_csv(os.path.join(COHA_SENSE_PROPS_DIR, f'{word}.csv'))
    df_sense_init_props = df_word_props[(df_word_props.year == 1810) & (df_word_props.prop_lexicon >= 0.000001)]
    return df_sense_init_props

def get_index_order(df_diffs, factor, is_ascending):
    """
    """
    n = len(df_diffs.index)
    L = df_diffs[factor].dropna().sort_values(ascending=is_ascending).index.to_list()
    L += (n - len(L)) * [np.nan]
    return np.array(L) + 1

### FACTOR EVALUATION

def eval_semantic_density(cur_sense, sense_nn, sense_embeds):
    """ Given a sense, return the average distance to its 10NN. 
    """
    cur_embed = sense_embeds[cur_sense]
    cur_word = get_word_from_sense(cur_sense)

    dists = []
    for i in range(10):
        sense = sense_nn[cur_sense][i]
        word = get_word_from_sense(sense)

        # ## Include only senses belonging to different words ## 
        # if word == cur_word:
        #     continue
        # ####

        dists.append(distance.cosine(cur_embed, sense_embeds[sense]))
    
    if len(dists) >= 7:
        return 1 - np.mean(dists)
    return np.nan

def eval_concreteness(cur_sense, sense_nn, conc_ratings):
    """ Given a sense, return the weighted mean of the concreteness ratings
    for 10NN, using word concreteness ratings by Kuperman 2013.
    Return NaN if 6 or less neighbours have a rating.
    """
    cur_word = get_word_from_sense(cur_sense)

    concs = []
    for i in range(10):
        sense = sense_nn[cur_sense][i]
        word = sense.split('_')[0]

        # ## Include only senses belonging to different words ## 
        # if word == cur_word:
        #     continue
        # ####

        # if word in conc_ratings:
        if sense in conc_ratings:
            concs.append(conc_ratings[sense])
    
    if len(concs) >= 7:  # 7 before
        return np.mean(concs) - 3  # Original scale is 1--5, center at 0
    return np.nan

def eval_valence(cur_sense, sense_nn, val_ratings):
    """ Given a sense, return the weighted mean of the valence ratings for 10NN, 
    using word concreteness ratings by Kuperman 2013.
    Return -10 if 6 or less neighbours have a rating.
    """
    cur_word = get_word_from_sense(cur_sense)

    vals = []
    for i in range(10):
        sense = sense_nn[cur_sense][i]
        word = sense.split('_')[0]

        # ## Include only senses belonging to different words ## 
        # if word == cur_word:
        #     continue
        # ####

        # if word in val_ratings:
        if sense in val_ratings:
            vals.append(np.abs(val_ratings[sense]))
    
    if len(vals) >= 7:  # 7 before
        return np.abs(np.mean(vals) - 5)  # Original scale is 1--9, center at 0, then get absolute valence
    return np.nan

def eval_sense_peripherality(cur_sense, sense_embeds, eval_type):
    """ Given a sense, return the average or sum distance to its fellow word
    senses. If the word has less than 2 senses, return NaN. 
    """
    if eval_type == 'sum':  # Depreciated! 
        return np.nan

    df_sense_init_props = get_sense_init_df(cur_sense)
    word_senses = df_sense_init_props['hu19_sense'].to_list()

    if len(word_senses) <= 1:
        # return 0
        return np.nan
    
    cur_embed = sense_embeds[cur_sense]

    dists = []
    for sense in word_senses:
        dists.append(distance.cosine(cur_embed, sense_embeds[sense]))

    return np.mean(dists)

def eval_sense_num(cur_sense):
    """ Given a sense, return the total number of senses (above the initial
    frequency threshold of 1/1M) for its word. 
    """
    df_sense_init_props = get_sense_init_df(cur_sense)
    word_senses = df_sense_init_props['hu19_sense'].to_list()

    return len(word_senses)

def eval_word_freq(cur_sense):
    """ Given a sense, return the initial (normalized) frequency of the word.
    """
    df_sense_init_props = get_sense_init_df(cur_sense)
    return sum(df_sense_init_props['prop_lexicon'])

def eval_sense_freq(cur_sense):
    """ Given a sense, return the initial (normalized) frequency of a sense.
    """
    df_sense_init_props = get_sense_init_df(cur_sense)
    return df_sense_init_props[['hu19_sense', 'prop_lexicon']].set_index('hu19_sense').to_dict()['prop_lexicon'][cur_sense]

def eval_contextual_div(cur_sense, eval_type):
    """ Given a sense, return the contextual diversity score (determined by
    eval_type) at time t=0 (ie. 1810). If no score exists, return -1. 
    """
    word = cur_sense.split('_')[0]
    cdiv_path = os.path.join(COHA_CDIV_DIR, f'{word}.csv')
    if not os.path.exists(cdiv_path):
        return np.nan
    df = pd.read_csv(cdiv_path)

    cdiv_val = df[(df['hu19_sense'] == cur_sense) & (df['year'] == 1810)][eval_type].item()
    if np.isnan(cdiv_val):
        return np.nan
    return cdiv_val

def eval_demand(cur_sense, sense_nn):
    """ Given a sense, for its 10NN, compute the mean value of the Spearman
    correlation between {1,2,...,20} and frequencies from t=1,2,...,20

    Omit senses which have a decline fit less than 0.1. 
    """
    comparison = list(range(1,21))
    cur_word = get_word_from_sense(cur_sense)

    corrs = []
    for i in range(10):
        sense = sense_nn[cur_sense][i]
        word = sense.split('_')[0]

        # ## Include only senses belonging to different words ## 
        # if word == cur_word:
        #     continue
        # ####

        df_word_props = pd.read_csv(os.path.join(COHA_SENSE_PROPS_DIR, f'{word}.csv'))
        freqs = df_word_props[df_word_props['hu19_sense'] == sense]['prop_lexicon'].to_list()

        if len(freqs) != 20:
            continue

        corrs.append(spearmanr(comparison, freqs).statistic)

    if len(corrs) >= 7:
        return np.mean(corrs)
    return np.nan

### MAIN

def get_factor_vals(matches_idx):
    """ Compute the values for each factor over each declining and stable sense.
    """
    factor_vals_path = os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_vals.pkl')
    if os.path.exists(factor_vals_path):
        return load_pickle(factor_vals_path)

    sense_matches = list(pd.read_csv(os.path.join(HU19_DIR, 'matches_10nn', f'hu19_sense_matches_{matches_idx}.csv')).to_records(index=False))

    sense_nn = load_pickle(os.path.join(HU19_DIR, 'sense_nn.pkl'))
    sense_embeds = load_pickle(os.path.join(HU19_DIR, 'diachronic_sense_emb.pkl'))
    # conc_ratings = pd.read_csv(os.path.join(KUPER13_DIR, 'concreteness_ratings.csv'))[['Word', 'Conc.M']].set_index('Word').to_dict()['Conc.M']
    conc_ratings = pd.read_csv(os.path.join(KUPER13_DIR, 'predictions', 'conc_no_both.csv'))[['sense', 'pred']].set_index('sense').to_dict()['pred']
    # val_ratings = pd.read_csv(os.path.join(KUPER13_DIR, 'valence_ratings.csv'))[['Word', 'V.Mean.Sum']].set_index('Word').to_dict()['V.Mean.Sum']
    val_ratings = pd.read_csv(os.path.join(KUPER13_DIR, 'predictions', 'val_no_both.csv'))[['sense', 'pred']].set_index('sense').to_dict()['pred']
    dec_metrics = pd.read_csv(os.path.join(COHA_SENSE_REGRESSION_DIR, 'decline-5.0.csv'))[['hu19_sense', 'decline_metric']].set_index('hu19_sense').to_dict()['decline_metric']

    factor_vals = {
        'sem_dens': {'dec': [], 'stb': []},
        'demand': {'dec': [], 'stb': []},
        's_perph': {'dec': [], 'stb': []},
        # 's_perph_sum': {'dec': [], 'stb': []},
        's_num': {'dec': [], 'stb': []},
        'conc': {'dec': [], 'stb': []},
        'val': {'dec': [], 'stb': []},
        'w_freq': {'dec': [], 'stb': []},
        # 's_freq': {'dec': [], 'stb': []},
        'c_div': {'dec': [], 'stb': []},
        # 'c_div_hull': {'dec': [], 'stb': []},
    }

    # Obtain all factor values
    for dec_sense, stb_sense in sense_matches:
        # Semantic Density
        factor_vals['sem_dens']['dec'].append(eval_semantic_density(dec_sense, sense_nn, sense_embeds))
        factor_vals['sem_dens']['stb'].append(eval_semantic_density(stb_sense, sense_nn, sense_embeds))

        # Demand
        # factor_vals['demand']['dec'].append(eval_demand(dec_sense, sense_nn))
        # factor_vals['demand']['stb'].append(eval_demand(stb_sense, sense_nn))

        if dec_metrics[dec_sense] <= 0.1:
            factor_vals['demand']['dec'].append(eval_demand(dec_sense, sense_nn))
            factor_vals['demand']['stb'].append(eval_demand(stb_sense, sense_nn))
        else:
            factor_vals['demand']['dec'].append(np.nan)
            factor_vals['demand']['stb'].append(np.nan)

        # Sense Peripherality (Mean)
        factor_vals['s_perph']['dec'].append(eval_sense_peripherality(dec_sense, sense_embeds, 'mean'))
        factor_vals['s_perph']['stb'].append(eval_sense_peripherality(stb_sense, sense_embeds, 'mean'))

        # Sense Peripherality (Sum)
        # factor_vals['s_perph_sum']['dec'].append(eval_sense_peripherality(dec_sense, sense_embeds, 'sum'))
        # factor_vals['s_perph_sum']['stb'].append(eval_sense_peripherality(stb_sense, sense_embeds, 'sum'))

        # Number of senses
        factor_vals['s_num']['dec'].append(eval_sense_num(dec_sense))
        factor_vals['s_num']['stb'].append(eval_sense_num(stb_sense))      

        # Concreteness
        factor_vals['conc']['dec'].append(eval_concreteness(dec_sense, sense_nn, conc_ratings))
        factor_vals['conc']['stb'].append(eval_concreteness(stb_sense, sense_nn, conc_ratings))

        # Valence
        factor_vals['val']['dec'].append(eval_valence(dec_sense, sense_nn, val_ratings))
        factor_vals['val']['stb'].append(eval_valence(stb_sense, sense_nn, val_ratings)) 

        # Word Frequency
        factor_vals['w_freq']['dec'].append(eval_word_freq(dec_sense))
        factor_vals['w_freq']['stb'].append(eval_word_freq(stb_sense))

        # Sense Frequency
        # factor_vals['s_freq']['dec'].append(eval_sense_freq(dec_sense))
        # factor_vals['s_freq']['stb'].append(eval_sense_freq(stb_sense))

        # Contextual Diversity (avg cost dist)
        factor_vals['c_div']['dec'].append(eval_contextual_div(dec_sense, 'avg_cos_dist'))
        factor_vals['c_div']['stb'].append(eval_contextual_div(stb_sense, 'avg_cos_dist'))

        # Contextual Diversity (convex hull dist)
        # factor_vals['c_div_hull']['dec'].append(eval_contextual_div(dec_sense, 'convex_hull_dist'))
        # factor_vals['c_div_hull']['stb'].append(eval_contextual_div(stb_sense, 'convex_hull_dist'))
    
    # If i is NaN in any factor, make it NaN in all factors. 
    # for i in range(len(sense_matches)):
    #     is_nan = False

    #     for factor in factor_vals:
    #         if np.isnan(factor_vals[factor]['dec'][i]) or np.isnan(factor_vals[factor]['stb'][i]):
    #             is_nan = True

    #     if is_nan:
    #         for factor in factor_vals:
    #             factor_vals[factor]['dec'][i] = np.nan
    #             factor_vals[factor]['stb'][i] = np.nan
    
    # If a NaN exists in a factor, set it to the mean value. 
    demand_dec_mean = np.nanmean(factor_vals['demand']['dec'])
    demand_stb_mean = np.nanmean(factor_vals['demand']['stb'])
    s_perph_dec_mean = np.nanmean(factor_vals['s_perph']['dec'])
    s_perph_stb_mean = np.nanmean(factor_vals['s_perph']['stb'])

    for i in range(len(sense_matches)):
        if np.isnan(factor_vals['demand']['dec'][i]):
            factor_vals['demand']['dec'][i] = demand_dec_mean
        if np.isnan(factor_vals['demand']['stb'][i]):
            factor_vals['demand']['stb'][i] = demand_stb_mean
        if np.isnan(factor_vals['s_perph']['dec'][i]):
            factor_vals['s_perph']['dec'][i] = s_perph_dec_mean
        if np.isnan(factor_vals['s_perph']['stb'][i]):
            factor_vals['s_perph']['stb'][i] = s_perph_stb_mean

    with open(factor_vals_path, 'wb') as out_f:
        pickle.dump(factor_vals, out_f)

    return factor_vals

def save_factor_vals(factor_vals, matches_idx):
    """ Save the factor values as csv's for dec and stb. Preserve match number. 
    """
    sense_matches = list(pd.read_csv(os.path.join(HU19_DIR, 'matches_10nn', f'hu19_sense_matches_{matches_idx}.csv')).to_records(index=False))
    
    dec_vals = {factor: factor_vals[factor]['dec'] for factor in factor_vals}
    df_dec = pd.DataFrame.from_dict(dec_vals)
    df_dec['sense'] = [pair[0] for pair in sense_matches]

    stb_vals = {factor: factor_vals[factor]['stb'] for factor in factor_vals}
    df_stb = pd.DataFrame.from_dict(stb_vals)
    df_stb['sense'] = [pair[1] for pair in sense_matches]

    cols = df_dec.columns.tolist()
    cols = cols[-1:] + cols[:-1]  # https://stackoverflow.com/a/13148611
    df_dec = df_dec[cols]
    df_stb = df_stb[cols]

    df_dec = df_dec.reset_index().rename(columns={'index': 'match_num'})
    df_stb = df_stb.reset_index().rename(columns={'index': 'match_num'})

    df_dec['match_num'] += 1
    df_stb['match_num'] += 1

    df_dec.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_dec.csv'), index=False)
    df_stb.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_stb.csv'), index=False)

def get_factor_diff(matches_idx, factor_vals):
    """ Return (and save) single-way (dec-stb) factor diffs, and also save
    the index rankings of pairs by the size of their difference. 
    """
    factor_diffs = {factor: np.array(factor_vals[factor]['dec']) - np.array(factor_vals[factor]['stb']) for factor in factor_vals}
    df_diffs = pd.DataFrame.from_dict(factor_diffs)
    df_diffs.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_diff.csv'), index=False)

    df_diffs['sem_dens'] = get_index_order(df_diffs, 'sem_dens', False)
    df_diffs['demand'] = get_index_order(df_diffs, 'demand', True)
    df_diffs['s_perph'] = get_index_order(df_diffs, 's_perph', False)
    df_diffs['s_num'] = get_index_order(df_diffs, 's_num', False)
    df_diffs['conc'] = get_index_order(df_diffs, 'conc', True)
    df_diffs['val'] = get_index_order(df_diffs, 'val', True)
    df_diffs['w_freq'] = get_index_order(df_diffs, 'w_freq', False)
    df_diffs['c_div'] = get_index_order(df_diffs, 'c_div', True)
    df_diffs.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_ranks.csv'), index=False)

    return factor_diffs

def get_factor_corr(matches_idx, factor_vals, factor_diffs):
    """ Compute and save correlation matrices for dec, stb, and diff value pairs
    for all factors. 
    """
    df_diffs = pd.DataFrame.from_dict(factor_diffs)
    df_corrs_diff = df_diffs.corr()
    df_corrs_diff.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_corr_diff.csv'), index=True)

    dec_vals = {factor: factor_vals[factor]['dec'] for factor in factor_vals}
    df_dec = pd.DataFrame.from_dict(dec_vals)
    df_corrs_dec = df_dec.corr()
    df_corrs_dec.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_corr_dec.csv'), index=True)

    stb_vals = {factor: factor_vals[factor]['stb'] for factor in factor_vals}
    df_stb = pd.DataFrame.from_dict(stb_vals)
    df_corrs_stb = df_stb.corr()
    df_corrs_stb.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_corr_stb.csv'), index=True)

def run_wilcoxon(matches_idx, factor_vals, factor_diffs):
    """ Run the Wilcoxon test on each factor, and save summary stats
    """
    stats = {
        "wilcoxon_p": [],
        "dec_mean": [],
        "dec_std": [],
        "stb_mean": [],
        "stb_std": [],
        "num": [],
    }

    for factor in factor_vals: 
        # diff = np.array(factor_vals[factor]['dec']) - np.array(factor_vals[factor]['stb'])
        diff = factor_diffs[factor][~np.isnan(factor_diffs[factor])]
        stats['wilcoxon_p'].append(wilcoxon(diff).pvalue)

        stats['dec_mean'].append(np.nanmean(factor_vals[factor]['dec']))
        stats['dec_std'].append(np.nanstd(factor_vals[factor]['dec']))
        stats['stb_mean'].append(np.nanmean(factor_vals[factor]['stb']))
        stats['stb_std'].append(np.nanstd(factor_vals[factor]['stb']))

        stats['num'].append(np.sum(~np.isnan(factor_vals[factor]['dec'])))

    stats['wilcoxon_p'] = sm.stats.multipletests(stats['wilcoxon_p'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)[1]

    # Save stats
    df_stats = pd.DataFrame.from_dict(stats, orient='index', columns=(list(factor_vals.keys()))).T
    df_stats = df_stats[['num', 'stb_mean', 'stb_std', 'dec_mean', 'dec_std', 'wilcoxon_p']]
    print(df_stats)
    # df_stats.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_wilcoxon_stats.csv'), index=True)

def run_logit(matches_idx, factor_vals):
    """ Run and save an analysis of the factor values using logistic regression.
    Specifically, given values for pairs of 'declining' and 'stable' senses,
    predict 1 for dec-stb, and 0 for stb-dec, and save the regression output. 
    """
    factor_diffs_half = {}

    ### Random indices implementation
    n_total = len(factor_vals['sem_dens']['dec'])
    n_half = n_total // 2

    for factor in factor_vals:
        # Random(8).shuffle(factor_vals[factor]['dec'])
        # Random(8).shuffle(factor_vals[factor]['stb'])

        first_diff = np.array(factor_vals[factor]['dec'])[:n_half] - np.array(factor_vals[factor]['stb'])[:n_half]
        second_diff = np.array(factor_vals[factor]['stb'])[n_half:] - np.array(factor_vals[factor]['dec'])[n_half:]
        
        factor_diffs_half[factor] = np.concatenate([first_diff, second_diff])

    factor_diffs_half['preds'] = np.concatenate([np.ones(n_half), np.zeros(n_half)])

    df_all = pd.DataFrame.from_dict(factor_diffs_half).dropna()

    df_preds = df_all.preds.squeeze()
    df_diffs = df_all.drop(columns=['preds'])

    df_diffs -= df_diffs.min() 
    df_diffs /= df_diffs.max()

    df_diffs = sm.add_constant(df_diffs)

    # ### Accuracy test
    # # https://medium.com/analytics-vidhya/cross-validation-in-machine-learning-using-python-4d0f335bec83
    # # https://www.geeksforgeeks.org/logistic-regression-using-statsmodels/
    # preds = []
    # for i in range(len(df_diffs.index)):
    #     df_train_preds, df_train_diffs = df_preds.drop([i]), df_diffs.drop([i])
    #     logit_res = sm.Logit(df_train_preds, df_train_diffs).fit()

    #     cur_pred = round(logit_res.predict(df_diffs.loc[[i]]))
    #     preds.append(cur_pred.to_list()[0] == df_preds.loc[[i]].to_list()[0])
    # print(sum(preds))
    # ### 

    logit_res = sm.Logit(df_preds, df_diffs).fit()
    # print(logit_res.summary())

    logit_sum0 = logit_res.summary().tables[0].as_csv()
    parse1 = [x.split(',') for x in logit_sum0.split('\n')]
    parse2 = [[item.strip(': ') for item in parse] for parse in parse1]
    parse3 = parse2[1:-1]
    parse4 = [[item[0], item[1]] for item in parse3] + [[item[2], item[3]] for item in parse3]

    df_logit_sum = pd.DataFrame(parse4, columns=['topic', 'value'])
    df_logit_sum.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_logit_sum.csv'), index=False)

    logit_sum1 = logit_res.summary().tables[1].as_csv()
    init_parse = [x.split(',') for x in logit_sum1.split('\n')]
    final_parse = [[item.strip() for item in parse] for parse in init_parse]
    final_parse[0][0] = 'factor'

    df_final = pd.DataFrame(final_parse[1:], columns=final_parse[0]).set_index('factor')
    df_final.to_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{matches_idx}', 'factor_analysis_logit_val.csv'), index=True)

def get_p_values_corr():
    """ For both the Wilcoxon and Logit methods, save a correlation matrix of 
    between the p-values of each trial. Report the mean and std of the upper 
    triangle of both as well.
    """
    p_wilcoxon = {}
    p_logit = {}

    for i in range(10):
        df_wilcoxon = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}', 'factor_analysis_wilcoxon_stats.csv'), index_col=0)
        df_logit = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}', 'factor_analysis_logit_val.csv'), index_col=0)
        df_logit = df_logit.drop(index=['const'])

        p_wilcoxon[i] = df_wilcoxon['wilcoxon_p'].to_numpy()
        p_logit[i] = df_logit['P>|z|'].to_numpy()

    # Save these
    df_wilcoxon_corrs = pd.DataFrame.from_dict(p_wilcoxon).corr()
    df_logit_corrs = pd.DataFrame.from_dict(p_logit).corr()

    df_wilcoxon_corrs.to_csv(os.path.join(COHA_SENSE_ANALYSIS_SUMMARY_DIR, 'wilcoxon_corrs.csv'))
    df_logit_corrs.to_csv(os.path.join(COHA_SENSE_ANALYSIS_SUMMARY_DIR, 'logit_corrs.csv'))

    # Save summary
    records = []
    for test_name, df in [('wilcoxon', df_wilcoxon_corrs), ('logit', df_logit_corrs)]:
        dists_mat = df.to_numpy()
        a, b = dists_mat.shape
        il1 = np.tril_indices(n=a, m=b, k=-1)  # Indices of lower triangle
        records.append((test_name, np.mean(dists_mat[il1]), np.std(dists_mat[il1])))
    
    df_recs = pd.DataFrame.from_records(records, columns=['test', 'mean', 'std'])
    df_recs.to_csv(os.path.join(COHA_SUMMARY56_DIR, 'hu19_factor_pval_corrs.csv'), index=False)

def get_p_values_var():
    """ Save the mean and standard deviation of each factor's p-value across
    the trials, for both tests.
    """
    p_wilcoxon = []
    p_logit = []

    for i in range(10):
        df_wilcoxon = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}', 'factor_analysis_wilcoxon_stats.csv'), index_col=0)
        df_logit = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}', 'factor_analysis_logit_val.csv'), index_col=0)
        df_logit = df_logit.drop(index=['const'])

        p_wilcoxon.append(df_wilcoxon['wilcoxon_p'].to_numpy())
        p_logit.append(df_logit['P>|z|'].to_numpy())

    factors = df_wilcoxon.index.to_list()
    num_factors = len(factors)

    p_wilcoxon = np.array(p_wilcoxon)
    p_logit = np.array(p_logit)

    pd.DataFrame(p_wilcoxon, columns=factors).to_csv(os.path.join(COHA_SENSE_ANALYSIS_SUMMARY_DIR, 'wilcoxon_ps.csv'))
    pd.DataFrame(p_logit, columns=factors).to_csv(os.path.join(COHA_SENSE_ANALYSIS_SUMMARY_DIR, 'logit_ps.csv'))

    records = []
    records += [('wilcoxon_mean', *[np.mean(p_wilcoxon[:, i]) for i in range(num_factors)])]
    records += [('wilcoxon_std', *[np.std(p_wilcoxon[:, i]) for i in range(num_factors)])]
    records += [('logit_mean', *[np.mean(p_logit[:, i]) for i in range(num_factors)])]
    records += [('logit_std', *[np.std(p_logit[:, i]) for i in range(num_factors)])]

    df_recs = pd.DataFrame.from_records(records, columns=(["factor"] + factors))
    df_recs.to_csv(os.path.join(COHA_SUMMARY56_DIR, 'hu19_factor_pval_var.csv'), index=False)

def get_beta_values():
    """ Save the beta values across trials
    """
    beta_logit = []

    for i in range(10):
        df_logit = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}', 'factor_analysis_logit_val.csv'), index_col=0)
        df_logit = df_logit.drop(index=['const'])

        beta_logit.append(df_logit['coef'].to_numpy())

    factors = df_logit.index.to_list()
    beta_logit = np.array(beta_logit)

    pd.DataFrame(beta_logit, columns=factors).to_csv(os.path.join(COHA_SENSE_ANALYSIS_SUMMARY_DIR, 'logit_betas.csv'))

def analyze_HU19_sense_factors():
    """ Run the Wilcoxon signed-rank test on our six factors: semantic density,
    concreteness, sense peripherality, word frequency, contextual diversity, 
    and demand.

    Report the mean and SD for each set of senses (dec and stb), as well as the
    p-value, for each factor.  
    """
    if not os.path.exists(COHA_SENSE_ANALYSIS_DIR):
        os.makedirs(COHA_SENSE_ANALYSIS_DIR)
    if not os.path.exists(COHA_SENSE_ANALYSIS_SUMMARY_DIR):
        os.makedirs(COHA_SENSE_ANALYSIS_SUMMARY_DIR)

    # Run the analysis for each trial, where i=0 is the null trial
    for i in tqdm(range(1)): # range(1)
        out_dir = os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Get factor vals
        factor_vals = get_factor_vals(i)

        save_factor_vals(factor_vals, i)

        # # Get factor diffs, then compute correlation
        factor_diffs = get_factor_diff(i, factor_vals)

        get_factor_corr(i, factor_vals, factor_diffs)

        # # Obtain summary and analysis stats using the Wilcoxon test
        run_wilcoxon(i, factor_vals, factor_diffs)

        # Obtain group factor analysis 
        run_logit(i, factor_vals)

    # Obtain the mean/std of the correlation between p-values from trials
    # get_p_values_corr()
    # get_p_values_var()
    # get_beta_values()


if __name__ == "__main__":
    analyze_HU19_sense_factors()
