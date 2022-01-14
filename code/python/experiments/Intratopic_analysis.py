from . import AbstractExperiment
import experimentalCollections as tc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import json
import pandas as pd
import convLib
from convLib import Annotator
import utils
import time
from retrieval.Analyzers import StandardAnalyzer

from convLib import contexts, Annotator

import experimentalCollections as EC

from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score

from scipy.stats import gaussian_kde, pearsonr
from scipy.spatial.distance import jensenshannon
from collections import Counter

import seaborn as sns
import random

from retrieval.CAsTutils import loadQueries
#'first_query', r"FU",
systems_order = ['context_query', 'linear_prev',  'RM3_prev', 'RM3_seq', 'allennlp_simple']
systems_labels = [ r"CU", r"LP", r"RM3p", r"RM3s", r"anCB"]

class Intratopic_analysis(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):
        sns.set_theme(style="whitegrid")
        #sns.set_context("paper",
        #                rc={"font.size": 24, "axes.titlesize": 24, "xtick.labelsize": 24, "ytick.labelsize": 24,
        #                     "legend.fontsize":24, "axes.labelsize": 24})

        measures = pd.read_csv(f"../../data/measures/full_valid_100.csv")

        avgs_all = measures\
                        [['topic', 'perm', 'qtype', 'name', 'nDCG@3']]\
                        .groupby(['topic', 'qtype', 'name', 'perm'])\
                        .aggregate(['mean'])\
                        .droplevel(1, axis=1)\
                        .reset_index() \

        avgs_all = avgs_all.groupby(['name', 'qtype', 'perm']).aggregate(['mean']).droplevel(1, axis=1).reset_index()

        avgs = measures[measures['name'].isin(systems_order)]\
                        [['topic', 'perm', 'qtype', 'name', 'nDCG@3']]\
                        .groupby(['topic', 'qtype', 'name', 'perm'])\
                        .aggregate(['mean'])\
                        .droplevel(1, axis=1)\
                        .reset_index()

        '''
        corrs = []
        for qtype in ['original', 'resolved']:
            avgs_t = avgs[avgs['qtype']==qtype][['name', 'topic', 'perm', 'nDCG@3']].copy()
            corr = avgs_t.groupby(['topic']).apply(rankingSimilarity)
            corr = pd.DataFrame([[qtype] + l for l in list(corr)], columns = ['qtype', 'topic', 'correlation', 'pvalue'])
            corrs.append(corr)
        corrs = pd.concat(corrs)
        fname = "../../data/correlation_valid_permutations.pdf"


        fig, ax = plt.subplots(figsize=(18, 12))
        h = sns.barplot(data=corrs, x='topic', y='correlation', hue='qtype', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:])
        ax.set(xlabel= "Conversation id", ylabel=r"Correlation (Spearman's $\rho$)")
        plt.savefig(fname)
        '''
        #for qtype in ['original', 'resolved']:
        for qtype in ['original']:
            avgs_t = avgs[(avgs['qtype']==qtype) & (avgs['topic'].isin(["32", "33", "37", "56", "58", "59", "61", "67", "69", "77", "78", "79"]))][['name', 'topic', 'perm', 'nDCG@3']].copy()
            Z = avgs_t.groupby(["topic"]).apply(pairwise_best).reset_index()
            print(Z)
            Z = Z.drop(columns=["level_1"])
            Z = Z.rename({"topic":"Conv. id"}, axis='columns')
            print(Z)
            grid = sns.FacetGrid(Z, col="Conv. id", col_wrap=4)
            grid.map(draw_heatmap, 's1', 's2', 'n_best', cbar=False, annot=True,  fmt='.3g')
            grid.set_titles(size=18)
            fname = f"../../data/figures/best_between_two_{qtype}.pdf"
            plt.savefig(fname)


def draw_heatmap(*args, **kwargs):
    data = pd.concat(args, axis=1)
    d = data.pivot(index=args[0].name, columns=args[1].name, values=args[2].name)
    d = d.sort_values('s1', key=make_sorter(systems_order))[systems_order]
    g = sns.heatmap(d, **kwargs)
    g.set_xticklabels(systems_labels, fontsize=18)
    g.set_yticklabels(systems_labels, fontsize=18)
    g.set_ylabel("")
    g.set_xlabel("")

def rankingSimilarity(scores, sim='pearson', index='name'):
    old_scores = scores[scores['perm']==0][['name', 'nDCG@3']]
    new_scores = scores[scores['perm']!=0][['name', 'nDCG@3']]

    temp_scores = pd.merge(old_scores, new_scores, how="inner", on=["name"])

    return [scores['topic'].unique()[0]] + list(pearsonr(temp_scores['nDCG@3_x'], temp_scores['nDCG@3_y']))


def pairwise_best(avgs_t):
    models = avgs_t['name'].unique()
    tmp = avgs_t.pivot_table(index='perm', columns='name', values='nDCG@3').reset_index()

    output=[[i, j, (np.sum(tmp[i]>tmp[j]))] for i in models for j in models if i!=j]
    output = pd.DataFrame(output, columns=["s1", "s2", "n_best"])

    return output

def make_sorter(l):
    """
    Create a dict from the list to map to 0..len(l)
    Returns a mapper to map a series to this custom sort order
    """
    sort_order = {k:v for k,v in zip(l, range(len(l)))}
    return lambda s: s.map(lambda x: sort_order[x])