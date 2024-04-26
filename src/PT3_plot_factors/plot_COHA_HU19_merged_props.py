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
PLOTS_MERGED_DIR = './results/COHA/plots/hidden1-maskedN/hu19_props_merged'


def plot_COHA_HU19_merged_props(dec_sense, stb_sense):
    """ Plot a declining and stable sense on the same plot, as well as the 
    other senses of the same word. 
    """
    dec_word, stb_word = dec_sense.split('_')[0], stb_sense.split('_')[0]

    df_dec_props = pd.read_csv(os.path.join(COHA_SENSE_PROPS_DIR, f'{dec_word}.csv'))
    df_stb_props = pd.read_csv(os.path.join(COHA_SENSE_PROPS_DIR, f'{stb_word}.csv'))
    df_props = pd.concat([df_dec_props, df_stb_props], ignore_index=True)

    dec_sense, stb_sense = 'language_3', 'marriage_1'
    df_props['hu19_sense'] = df_props['hu19_sense'].replace({
        'language_1_noun_1': 'language_1', 
        'language_1_noun_2': 'language_2', 
        'language_1_noun_3': 'language_3', 
        'marriage_1_noun_1': 'marriage_1',
        'marriage_1_noun_2': 'marriage_2'
    })

    # print(df_props)

    df_dec = df_props[df_props['hu19_sense'] == dec_sense]
    df_stb = df_props[df_props['hu19_sense'] == stb_sense]

    df_rest = df_props[~df_props['hu19_sense'].isin([dec_sense, stb_sense])]

    df_dec_rest = df_rest[df_rest['hu19_sense'].str.split('_', expand=True)[0] == dec_word]
    df_stb_rest = df_rest[df_rest['hu19_sense'].str.split('_', expand=True)[0] == stb_word]

    # https://stackoverflow.com/a/58432483
    # https://stackoverflow.com/a/33514939

    palette_name = "magma_r"  # https://seaborn.pydata.org/tutorial/color_palettes.html
    light, dark = sns.color_palette(palette_name)[1], sns.color_palette(palette_name)[-3]  

    # Main lines
    sns_plot = sns.lineplot(
        data=df_dec,
        x="year", y="prop_lexicon", hue="hu19_sense", style="hu19_sense",
        dashes=[(2, 1)], 
        palette=[light], 
        linewidth=4, zorder=5,
        # kind="line",
        # legend=False,
    )

    sns.lineplot(
        data=df_stb,
        x="year", y="prop_lexicon", hue="hu19_sense", style="hu19_sense",
        dashes=False, 
        palette=[dark], 
        linewidth=4, zorder=5,
        # kind="line",
        # legend=False,
    )

    # Accessory lines
    sns.lineplot(
        data=df_dec_rest, 
        x="year", y="prop_lexicon", hue="hu19_sense", style="hu19_sense",
        dashes=[(3, 2), (1, 1)], #  [(2, 1)] * df_dec_rest['hu19_sense'].nunique(), 
        palette=[light] * df_dec_rest['hu19_sense'].nunique(), 
        linewidth=2,
        # legend=False,
    )

    sns.lineplot(
        data=df_stb_rest, 
        x="year", y="prop_lexicon", hue="hu19_sense", style="hu19_sense",
        dashes=False, 
        palette=[dark] * df_stb_rest['hu19_sense'].nunique(), 
        linewidth=2,
        # legend=False,
    )

    # Visual formatting
    for line in sns_plot.legend().get_lines():
        # print(line.get_label())
        if line.get_label() in [dec_sense, stb_sense]:
            line.set_linewidth(4.0)

    # sns_plot.yaxis.grid(True) # Hide the horizontal gridlines
    # sns_plot.xaxis.grid(False) # Show the vertical gridlines
        
    sns_plot.margins(x=0, y=0.01)  # https://stackoverflow.com/a/45880976
    sns_plot.figure.tight_layout()

    # Set axis names
    sns_plot.set(xlabel='Year (decade)', ylabel='Relative frequency')

    sns_plot.figure.savefig(os.path.join(PLOTS_MERGED_DIR, f'{dec_word}_{stb_word}.pdf'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', type=str, help='declining sense', default='language_1_noun_3')
    parser.add_argument('-S', type=str, help='stable sense', default='marriage_1_noun_1')

    args = parser.parse_args()

    plot_COHA_HU19_merged_props(args.D, args.S)
