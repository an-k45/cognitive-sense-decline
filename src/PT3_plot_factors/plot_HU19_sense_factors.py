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

import seaborn as sns

sns.set_style("darkgrid")

COHA_SENSE_ANALYSIS_DIR = './data/COHA/hidden1-maskedN/sense_analysis_all10nn_matnanmean'
PLOTS_SENSE_ANALYSIS_CORR_DIR = './results/COHA/plots/hidden1-maskedN/sense_analysis_all10nn_matnanmean/corr'
PLOTS_SENSE_ANALYSIS_DIFF_DIR = './results/COHA/plots/hidden1-maskedN/sense_analysis_all10nn_matnanmean/diff'

def load_pickle(f_path):
    with open(f_path, 'rb') as in_f:
        data = pickle.load(in_f)
    return data

def adjust_diffs_from_vals(factor_vals):
    """ Given the dict of factor_vals, normalize them using Z-scores, and return
    the difference. 
    """
    factor_diffs = {}
    for factor in factor_vals:
        z_dec = (np.array(factor_vals[factor]['dec']) - np.nanmean(factor_vals[factor]['dec'])) / np.nanstd(factor_vals[factor]['dec'])
        z_stb = (np.array(factor_vals[factor]['stb']) - np.nanmean(factor_vals[factor]['stb'])) / np.nanstd(factor_vals[factor]['stb'])
        factor_diffs[factor] = z_dec - z_stb
    df_diffs = pd.DataFrame.from_dict(factor_diffs)
    return df_diffs

def adjust_diffs(df):
    """ 
    """
    for col in df:
        df[col] = (df[col] - np.mean(df[col])) / np.std(df[col])
    return df

def plot_diffs(df, matches_idx):
    """ Plot diffs using box plots
    """
    out_path = os.path.join(PLOTS_SENSE_ANALYSIS_DIFF_DIR, f'factor_analysis_diffs_{matches_idx}.png')

    sns_plot = sns.boxplot(data=pd.melt(df), x="value", y="variable", orient='h')
    sns_plot.figure.tight_layout()
    sns_plot.figure.savefig(out_path)
    sns_plot.figure.clf()

def plot_corrs(df, matches_idx):
    """ Plot correlations using a heatmap
    """
    out_path = os.path.join(PLOTS_SENSE_ANALYSIS_CORR_DIR, f'factor_analysis_corr_{matches_idx}.png')

    plot_cmap = sns.color_palette("vlag", as_cmap=True)  # https://seaborn.pydata.org/tutorial/color_palettes.html#custom-diverging-palettes

    sns_plot = sns.heatmap(data=df, vmin=-1, vmax=1, annot=True, annot_kws={"size": 11}, fmt=".2f", cmap=plot_cmap)
    sns_plot.figure.tight_layout()
    sns_plot.figure.savefig(out_path)
    sns_plot.figure.clf()

def plot_HU19_sense_factors():
    """
    """
    if not os.path.exists(PLOTS_SENSE_ANALYSIS_CORR_DIR):
        os.makedirs(PLOTS_SENSE_ANALYSIS_CORR_DIR)

    if not os.path.exists(PLOTS_SENSE_ANALYSIS_DIFF_DIR):
        os.makedirs(PLOTS_SENSE_ANALYSIS_DIFF_DIR)

    for i in tqdm(range(10)):
        # df_diffs = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}', 'factor_analysis_diff.csv'))
        # df_diffs = adjust_diffs(df_diffs)
        factor_vals = load_pickle(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}', 'factor_analysis_vals.pkl'))
        df_diffs = adjust_diffs_from_vals(factor_vals)

        df_corrs = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_DIR, f'matches_{i}', 'factor_analysis_corr_diff.csv'), index_col=0)
        # df_corrs = df_corrs.rename({"sem_dens": })

        plot_diffs(df_diffs, i)
        plot_corrs(df_corrs, i)


if __name__ == "__main__":
    plot_HU19_sense_factors()
