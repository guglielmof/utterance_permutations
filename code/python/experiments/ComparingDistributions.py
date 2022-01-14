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

from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from collections import Counter

import seaborn as sns
import random

from retrieval.CAsTutils import loadQueries
#'first_query',r"FU",
systems_order = ['context_query', 'linear_prev',  'RM3_prev', 'RM3_seq', 'allennlp_simple']
systems_labels = [r"CU", r"LP", r"RM3p", r"RM3s", r"anCB"]
class ComparingDistributions(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):

        sns.set_theme(style="whitegrid")
        sns.set_context("paper",
                        rc={"font.size": 30, "axes.titlesize": 30, "xtick.labelsize": 30, "ytick.labelsize": 30,
                             "legend.fontsize":30, "axes.labelsize": 30})



        measures = pd.read_csv(f"../../data/measures/full_valid_100.csv")

        avgs_all = measures\
                        [['topic', 'perm', 'qtype', 'name', 'nDCG@3']]\
                        .groupby(['topic', 'qtype', 'name', 'perm'])\
                        .aggregate(['mean'])\
                        .droplevel(1, axis=1)\
                        .reset_index() \

        avgs_all = avgs_all.groupby(['name', 'qtype', 'perm']).aggregate(['mean']).droplevel(1, axis=1).reset_index()


        print(avgs_all[avgs_all['perm']==0][['name', 'qtype', 'nDCG@3']].pivot_table(index='name', columns='qtype', values='nDCG@3'))




        avgs = measures[~measures['name'].isin(['BM25', 'RM3', 'DirichletLM'])]\
                        [['topic', 'perm', 'qtype', 'name', 'nDCG@3']]\
                        .groupby(['topic', 'qtype', 'name', 'perm'])\
                        .aggregate(['mean'])\
                        .droplevel(1, axis=1)\
                        .reset_index()





        print(avgs.groupby(['name', 'qtype']).aggregate(['mean']).reset_index()[['name', 'qtype', 'nDCG@3']].pivot_table(index='name', columns='qtype', values='nDCG@3'))


        for qtype in ['original']:
            # & (avgs['topic'].isin(["32", "37", "40", "54", "59", "61", "67", "68", "69", "77", "78", "79"]))

            avgs_t = avgs[(avgs['qtype']==qtype) & (avgs['topic'].isin(["32", "33", "37", "56", "58", "59", "61", "67", "69", "77", "78", "79"]))][['name', 'topic', 'perm', 'nDCG@3']]
            plot_boxplot(avgs_t, f"../../data/figures/boxplots_valid_permutations_{qtype}.pdf")
            #plot_bestModel(avgs_t, f"../../data/n_times_best_system_{qtype}.pdf")
            print_largest_distance(avgs_t)



def print_largest_distance(avgs):
    bestPerms = avgs.pivot_table('nDCG@3', ['topic', 'perm'], 'name').reset_index().groupby(['topic']).apply(
        lambda x: getBestPerms(x, list(avgs['name'].unique()))).reset_index().drop("level_1", axis=1)

    bperf = bestPerms.groupby(['name', 's2']).apply(lambda x: getBestPerformance(x, avgs))

    bperf = bperf.reset_index()

    bperf = bperf[['name', 's2', 'name_y', 'nDCG@3']]

    df1 = bperf[(bperf['s2']==bperf['name_y']) & (bperf['s2']!=bperf['name'])]
    df2 = bperf[(bperf['name']==bperf['name_y']) & (bperf['s2']!='mean')]

    df3 = bperf[(bperf['s2']=='mean') & (bperf['name']!=bperf['name_y'])]
    df4 = bperf[(bperf['s2']=='mean') & (bperf['name']==bperf['name_y'])]

    df3 = pd.merge(df3, df4, how="inner", on=["name", 's2'])
    df3['diff'] = df3['nDCG@3_y'] - df3['nDCG@3_x']
    df3 = df3[['name', 'diff']].groupby(['name']).agg(np.mean).reset_index()
    with pd.option_context('display.max_rows', None, 'display.max_columns', bperf.shape[1]):
        print(df3)



    bperf = pd.merge(df1, df2, how="inner", on=["name", 's2'])[['name', 's2', 'nDCG@3_x', 'nDCG@3_y']]

    bperf['diff'] = bperf['nDCG@3_y'] - bperf['nDCG@3_x']

    bperf = bperf.pivot_table(index='name', columns='s2', values='diff')
    with pd.option_context('display.max_rows', None, 'display.max_columns', bperf.shape[1]):
        print(bperf)
        
        


def getBestPerformance(ds, originalPerformance):
    tmpds = ds.merge(originalPerformance, how='inner', on=['perm', 'topic'])

    return (tmpds.groupby('name_y').agg(np.mean))



def getBestPerms(ds, systems):
    values = []
    for s1 in systems:

        values.append(ds[s1].values)

    values = np.array(values)

    result = []

    for e1, s1 in enumerate(systems):
        diffs = []
        for e2, s2 in enumerate(systems):
            diff = values[e2] - values[e1]

            diffs.append(diff)
            result.append([s1, s2, np.argmin(diff)])

        result.append([s1, 'mean', np.argmin(np.mean(diffs, axis=0))])

    result = pd.DataFrame(result, columns=['name', 's2', 'perm'])
    return result


def plot_boxplot(avgs, fname="../../data/figures/boxplots_valid_permutations.pdf"):
    plt.figure(figsize=(24, 12))
    width = 0.15
    systems = list(avgs['name'].unique())
    ns = len(systems)
    plot = sns.boxplot(data=avgs, x='topic', y='nDCG@3', hue='name', width=ns * width, hue_order=systems_order)

    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

    # colors = ['k', 'r', 'g', 'b', 'y']


    for e, nm in enumerate(systems_order):
        offset = ((e) - (ns // 2)) * width
        offset = offset + width/2

        x = np.arange(len(avgs['topic'].unique())) + offset
        sns.scatterplot(x=x,
                        y=avgs[(avgs["name"] == nm) & (avgs['perm'] == 0)]["nDCG@3"],
                        color='y',
                        edgecolor='k',
                        marker="d", zorder=10, s=100)

    handles, labels = plot.axes.get_legend_handles_labels()
    plot.axes.get_legend().remove()
    plt.legend(handles, systems_labels, ncol=len(labels), fontsize=30, loc='upper center',
                    bbox_to_anchor=(0.48, 1.1), frameon=False)
    plot.axes.set(xlabel="Conversation id")
    plt.savefig(fname)

def plot_bestModel(avgs, fname="../../data/figures/n_times_best_system.pdf"):
        bestModel = avgs.loc[avgs.groupby(['topic', 'perm'])['nDCG@3'].idxmax()]
        bestModel = bestModel.groupby(["topic", "name"]).agg({'name':'count'}).rename({'name':'count'}, axis=1).reset_index()
        plt.figure(figsize=(18, 12))

        plot = sns.barplot(data=bestModel, hue='name', x='topic', y='count', hue_order=systems_order)
        handles, labels = plot.axes.get_legend_handles_labels()
        plot.axes.get_legend().remove()
        plt.legend(handles, systems_labels, ncol=len(labels), fontsize=18, loc='upper center',
                   bbox_to_anchor=(0.48, 1.1), frameon=False)
        plot.axes.set(xlabel="Conversation id")

        plt.savefig(fname)