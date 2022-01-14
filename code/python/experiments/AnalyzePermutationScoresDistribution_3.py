from . import AbstractExperiment
import experimentalCollections as tc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from matplotlib import cm
import json
import pandas as pd
import convLib
from convLib import Annotator
import utils
import time
from retrieval.Analyzers import StandardAnalyzer
from itertools import chain
from convLib import contexts, Annotator


import experimentalCollections as EC

from sklearn.preprocessing import normalize

from collections import Counter

import seaborn as sns

class AnalyzePermutationScoresDistribution_3(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        measures = pd.read_csv(f"../../data/measures/full_valid_100.csv")


        avgs = measures[(~measures['name'].isin(['BM25', 'RM3', 'DirichletLM', 'first_query'])) & (measures['qtype']=='original')]\
                        [['topic', 'perm', 'qtype', 'name', 'nDCG@3']]\
                        .groupby(['topic', 'qtype', 'name', 'perm'])\
                        .aggregate(['mean'])\
                        .droplevel(1, axis=1)\
                        .reset_index()

        avgs_topics = avgs[['topic', 'perm', 'nDCG@3']].groupby(['topic', 'perm']).aggregate(['mean']).droplevel(1,
                                                                                                                 axis=1).reset_index()
        sns.set_theme(style="whitegrid")
        sns.set_context("paper",
                        rc={"font.size": 30, "axes.titlesize": 30, "xtick.labelsize": 30, "ytick.labelsize": 30,
                             "legend.fontsize":30, "axes.labelsize": 30})
        plt.figure(figsize=(24, 12))
        sns.boxplot(data=avgs_topics.rename({'topic':'Conv. id'}, axis='columns'), x='Conv. id', y='nDCG@3')
        diamonds = avgs_topics[avgs_topics['perm'] == 0]
        diamonds_y = np.array(diamonds['nDCG@3'])

        plt.scatter(x=np.arange(len(diamonds_y)), y=diamonds_y,
                    color='y',
                    edgecolor='k',
                    marker="d", zorder=10, s=300)

        plt.savefig(f"../../data/figures/boxplots_valid_permutations_means_original.pdf")

        B=100000
        unique_perms = list(avgs['perm'].unique())

        sampled_means = []
        topics = list(avgs['topic'].unique())
        for b in range(B):

            # sampling I
            sampled_perms = npr.choice(unique_perms, len(topics))
            #ser = pd.Series(sampled_perms)

            #idx = ser.map({k: avgs.index[avgs['perm'].values == k] for k, v in ser.value_counts().items()})



            # sampling II

            #ser = pd.Series(sampled_perms)
            #idx = ser.map({k[0]: avgs.index[(avgs['perm'].values == k[1]) & (avgs['topic'].values == topics[e])] for e, k in enumerate(ser.value_counts().items())})

            #df_new = avgs.loc[list(chain.from_iterable(idx))].reset_index(drop=True)
            df_new = pd.concat([avgs[(avgs['perm'].values == i) & (avgs['topic'].values == topics[e])] for e, i in enumerate(sampled_perms)])
            sampled_means.append(df_new[['name', 'nDCG@3']].groupby(['name']).mean().reset_index())

        sampled_means = pd.concat(sampled_means)


        print(sampled_means.groupby(['name']).aggregate([lambda x: np.quantile(x, 0.025), lambda x: np.quantile(x, 0.975), np.min, np.max]))
