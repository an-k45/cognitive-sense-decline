This repository contains code and data for Kali et al. (2024), "Cognitive Factors in Word Sense Decline".

## Code
The code to run this project is organized in 3 portions (+1 setup portion), all under /src.

Part 0 (data & setup): Scripts relevant to setting up our data for analysis.
 - `parse_COHA.py`: Given raw COHA data (Alatrash et al. 2020; Davies et al. 2012), split it into .csv's containing all sentences per lemma
 - `get_COHA_embeddings_all.py`: Compute contextual usage BERT embeddings for the 3220 words from Hu et al. (2019) (henceforth HU19)
 - `get_COHA_HU19_classification.py`: Classify all usage embeddings to a HU19 defined sense 
 - `summarize_COHA_temporal_counts.py`: Produce per-decade summary stats on COHA
 - `summarize_COHA_HU19_words.py`: Produced per-word summary stats on the HU19 word list
 - `get_COHA_HU19_sense_props.py`: Count and compute per-decade proportions for senses at the word-level and lexicon-level

Part 1 (analysis setup): Scripts relevant to set up various parts of the main analysis
 - `get_COHA_HU19_sense_regression.py`: Identify declining and stable senses
 - `get_HU19_sense_nn.py`: Compute all NN for all senses in HU19
 - `get_HU19_sense_matchings.py`: Match pairs of declining and stable senses
 - `get_HU19_concval_ratings.py`: Predict missing concreteness and valence ratings

Part 2 (analysis factors): Scripts to actually run the factor analysis
 - `analyze_HU19_sense_factors.py`: Analyze factors using Wilcoxon and Logit tests
 - `query_HU19_sense_example.py`: Query information about a single sense, including factor data 

Part 3 (plotting results): Scripts to plot various results for presentation
 - `plot_COHA_HU19_sense_props.py`: Plot per-decade proportions for each sense, per word. 
 - `plot_HU19_sense_factors.py`: Plot correlation matrices and z-score distributions for factor differences
 - `plot_HU19_sense_analysis_summary.py`: Plot per-trial p-values, Beta values, and correlations

## Data and Results
Data and results are shared to the extent possible.

Data (./data):
 - /hu2019: Set of matches, and definitions from their published data. Embeddings were obtained by contacting the authors.
 - /COHA: List of declining and stable senses, and full results for all trials in the factor analysis. For results in the paper, see under 'matches_0': "factor_analysis_wilcoxon_stats.csv" and "factor_analysis_logit_val.csv".

Results (./results):
 - /plots: Per-decade plots for words (at the word- and lexicon-level). Correlations, Z-score distributions, and per-trial summaries, for all of the factor analysis. 
 - /stats: Summary stats on our set of words, matches, and factor analysis trials.
