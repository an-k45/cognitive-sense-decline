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

import seaborn as sns

sns.set_style("darkgrid")

COHA_SENSE_PROPS_DIR = './data/COHA/hidden1-maskedN/hu19_props'
PLOTS_WORDS_DIR = './results/COHA/plots/hidden1-maskedN/hu19_props_word'
PLOTS_LEXICON_DIR = './results/COHA/plots/hidden1-maskedN/hu19_props_lexicon'


def setup_dirs(target_dir):
    """ Remove existing data and create empty folders
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def plot_time_proportions(word, df_sense_props, prop_type, prop_name, plots_dir):
    """
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    """
    out_path_png = os.path.join(plots_dir, 'png', f'{word}.png')
    out_path_pdf = os.path.join(plots_dir, 'pdf', f'{word}.pdf')

    sns_plot = sns.lineplot(
        data=df_sense_props, 
        x="year", y=prop_type, 
        hue="hu19_sense", style="hu19_sense", markers=True, dashes=False
    )
    
    # box = sns_plot.get_position()
    # sns_plot.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])
    
    tlt = sns_plot.figure.suptitle(f'{word}')
    lgd = sns_plot.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.125), 
        ncol=3
    )  # https://stackoverflow.com/a/4701285
    
    sns_plot.margins(x=0.01)  # https://stackoverflow.com/a/45880976
    sns_plot.set_ylabel(prop_name)

    # sns_plot.figure.tight_layout()
    sns_plot.figure.savefig(out_path_png, bbox_extra_artists=(lgd,tlt), bbox_inches='tight')  # https://stackoverflow.com/a/10154763
    sns_plot.figure.savefig(out_path_pdf, bbox_extra_artists=(lgd,tlt), bbox_inches='tight')  # https://stackoverflow.com/a/10154763
    sns_plot.figure.clf()

def plot_COHA_HU19_sense_props():
    """ Save a plot of the proportion of senses in the word, and in the lexicon.
    """
    out_dirs = [os.path.join(PLOTS_WORDS_DIR, 'png'), os.path.join(PLOTS_WORDS_DIR, 'pdf'), os.path.join(PLOTS_LEXICON_DIR, 'png'), os.path.join(PLOTS_LEXICON_DIR, 'pdf')]
    for out_dir in out_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    for f_word in tqdm(os.listdir(COHA_SENSE_PROPS_DIR)):
        word = os.path.splitext(f_word)[0]

        coha_sense_props_path = os.path.join(COHA_SENSE_PROPS_DIR, f'{word}.csv')
        df_sense_props = pd.read_csv(coha_sense_props_path)

        plot_time_proportions(word, df_sense_props, 'prop_word', 'word%', PLOTS_WORDS_DIR)
        plot_time_proportions(word, df_sense_props, 'prop_lexicon', 'lexicon%', PLOTS_LEXICON_DIR)


if __name__ == "__main__":
    plot_COHA_HU19_sense_props()
