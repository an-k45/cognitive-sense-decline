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

COHA_SENSE_ANALYSIS_SUM_DIR = './data/COHA/hidden1-maskedN/sense_analysis_summary_all10nn_matnanmean'
PLOTS_SENSE_ANALYSIS_SUM_DIR = './results/COHA/plots/hidden1-maskedN/sense_analysis_all10nn_matnanmean/summary'


def plot_heat(df, f_name, v1=-1, v2=1):
    """ Plot values using a heatmap
    """
    out_path = os.path.join(PLOTS_SENSE_ANALYSIS_SUM_DIR, f_name)

    plot_cmap = sns.color_palette("vlag", as_cmap=True)  # https://seaborn.pydata.org/tutorial/color_palettes.html#custom-diverging-palettes

    sns_plot = sns.heatmap(data=df, vmin=v1, vmax=v2, annot=True, fmt=".2f", cmap=plot_cmap)
    sns_plot.figure.tight_layout()
    sns_plot.figure.savefig(out_path)
    sns_plot.figure.clf()

def plot_HU19_sense_analysis_summary():
    """
    """
    if not os.path.exists(PLOTS_SENSE_ANALYSIS_SUM_DIR):
        os.makedirs(PLOTS_SENSE_ANALYSIS_SUM_DIR)

    df_wilcoxon_corrs = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_SUM_DIR, 'wilcoxon_corrs.csv'), index_col=0)
    df_logit_corrs = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_SUM_DIR, 'logit_corrs.csv'), index_col=0)

    plot_heat(df_wilcoxon_corrs, 'wilcoxon_corrs.png')
    plot_heat(df_logit_corrs, 'logit_corrs.png')

    df_wilcoxon_ps = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_SUM_DIR, 'wilcoxon_ps.csv'), index_col=0)
    df_logit_ps = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_SUM_DIR, 'logit_ps.csv'), index_col=0) 

    plot_heat(df_wilcoxon_ps, 'wilcoxon_ps.png')
    plot_heat(df_logit_ps, 'logit_ps.png')

    df_logit_betas = pd.read_csv(os.path.join(COHA_SENSE_ANALYSIS_SUM_DIR, 'logit_betas.csv'), index_col=0)

    plot_heat(df_logit_betas, 'logit_betas.png', v1=-10, v2=10)


if __name__ == "__main__":
    plot_HU19_sense_analysis_summary()
