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


class UtterancesSampling(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):
        df = pd.read_csv("../../data/measures/permutated_conversations_resolved_100.csv")
        df[['topic', 'utterance']] = df['qid'].str.split("_", 1, expand=True)

        df = df.drop(['Unnamed: 0', 'Unnamed: 0.1',
                      'AP', 'P@1', 'P@3', 'RR@10', 'nDCG@10',
                      ], axis=1)\


        df.columns = ["model", "qid", "score", "perm", "order", "topic", "utterance"]



        perms = {t:
                     {p: df[(df['perm'] == p) & (df['topic'] == t) & (df['model'] == 'RM3_prev')].sort_values('order',
                                                                                                             ascending=True)[
                         'qid'].values
                      for p in df['perm'].unique()}
                 for t in df['topic'].unique()}



        B = 1000

        means = df[~df['model'].isin(['BM25', 'RM3'])]\
                    [['model', 'topic', 'perm', 'score']]\
                    .groupby(['model', 'topic', 'perm'])\
                    .aggregate(["mean"])\
                    .reset_index()\
                    .droplevel(1, axis=1)


        refDistr = means\
                    .groupby(["model", "topic"])\
                    .apply(getKDE)\
                    .reset_index()\
                    .rename({0: 'refdistr'}, axis='columns')


        #bootstrapEval = exactBootstrap(means, B=B)

        #bootstrapEval = deltaBootstrap(df, B=B)

        #bootstrapEval = refBootstrap(df, B=B, perms=perms)

        #bootstrapEval = multiplicativeBootstrap(df, B=B)

        bootstrapEval = mixedModelsBootstrap(df, perms, B=B)
        evaluate(bootstrapEval, refDistr)


def getKDE(x, col="score"):
    distr = lambda x: np.zeros(len(x))+0.0001
    try:
        distr = gaussian_kde(x[col].values)
    except Exception as e:
        print(e)
        print(f"model: {x['model'].values[0]} conv: {x['topic'].values[0]}")
    finally:
        return distr

def evaluate(ds, refDistr, plotDistr=False):
    plt.figure(figsize=(20, 11))

    sns.boxplot(data=ds, x='topic', y='score', hue='model', width=len(ds['model'].unique()) * 0.15)

    smpDistr = ds.groupby(['model', 'topic'])\
                 .apply(getKDE) \
                 .reset_index() \
                 .rename({0: 'smpdistr'}, axis='columns')

    distrs = pd.concat([refDistr, smpDistr], axis=1, join='inner')
    distrs = distrs.loc[:, ~distrs.columns.duplicated()]

    X = np.linspace(0, 1, 100)
    jsd = lambda x: jensenshannon(x['smpdistr'](X), x['refdistr'](X))
    distrs['jsd'] = distrs.apply(jsd, axis=1)

    print(distrs[["model", "jsd"]].groupby(["model"]).aggregate(["mean"]))
    print(distrs[["topic", "jsd"]].groupby(["topic"]).aggregate(["mean"]))
    print(f"avg jsd: {np.mean(distrs[distrs['jsd'].notna()]['jsd'].values):.4f} (na. {len(distrs[~distrs['jsd'].notna()]['jsd'].values)})")

    def draw_lc(*args, **kwargs):
        data = kwargs.pop('data')
        temp_ds = []
        for s in sorted(data['model'].unique()):
            temp_ds += [[s, 'smpdistr', x,  v] for x, v in zip(X, data[data['model'] == s]['smpdistr'].values[0](X))]
            temp_ds += [[s, 'refdistr', x,  v] for x, v in zip(X, data[data['model'] == s]['refdistr'].values[0](X))]


        temp_ds = pd.DataFrame(temp_ds, columns = ['model', 'distribution', 'X', 'Y'])
        sns.lineplot(data=temp_ds, x='X', y='Y', hue='model', style='distribution')

    fg = sns.FacetGrid(distrs, col='topic', col_wrap=5, sharex=False, sharey=False)
    fg.map_dataframe(draw_lc, 'topic', 'refdistr', 'smpdistr', cbar=False)

    plt.figure(figsize=(20, 11))

    sns.histplot(data=distrs, x='jsd', hue='model', multiple="stack")


def exactBootstrap(means, B=10000):

    bootstrapSamples = []
    for s in sorted(means['model'].unique()):
        for t in sorted(means['topic'].unique()):
            vals = means[(means['model']==s)&(means['topic']==t)]['score'].values
            bootstrapSamples += [[s, t, i, random.choice(vals)] for i in range(B)]

    bootstrapSamples = pd.DataFrame(bootstrapSamples, columns=['model', 'topic', 'k', 'score'])
    return bootstrapSamples


def multiplicativeBootstrap(df, B=10000):
    refScore = df[(~df['model'].isin(['BM25', 'RM3'])) & (df['order']==0)]\
              [['model', 'qid', 'score']]\
              .groupby(["model", "qid"])\
              .aggregate(["mean"])\
              .reset_index()\
              .droplevel(1, axis=1)


    diffDf = df[(~df['model'].isin(['BM25', 'RM3'])) & (df['perm'] == 0)].copy()

    diffDf['prev_score'] = diffDf.apply(lambda x: refScore[(refScore['model']==x['model'])&(refScore['qid']==x['qid'])]['score'].values[0], axis=1)

    diffDf['diffs'] = diffDf['score'] - diffDf['prev_score']


    bootstrapSamples = _boostsrapDiffs(diffDf, B)

    return bootstrapSamples


def mixedModelsBootstrap(df, perms, B=10000):

    refScore = df[(~df['model'].isin(['BM25', 'RM3'])) & (df['order']==0)]\
              [['model', 'qid', 'score']]\
              .groupby(["model", "qid"])\
              .aggregate(["mean"])\
              .reset_index()\
              .droplevel(1, axis=1)

    refDict = {}
    for i, r in refScore.iterrows():
        if r['model'] not in refDict:
            refDict[r['model']] = {}
        refDict[r['model']][r['qid']] = r['score']

    def getPreviousUtterance(x):
        return perms[x['topic']][x['perm']][max(x['order'] - 1, 0)]

    #######
    df['prev'] = df.apply(getPreviousUtterance, axis=1)
    df[['prev_topic', 'prev_utterance']] = df['prev'].str.split("_", 1, expand=True)


    pairwise_difference = {}
    for i, r in df[~df['model'].isin(["BM25", "RM3"])].iterrows():
        if r['model'] not in pairwise_difference:
            pairwise_difference[r['model']] = {}
        if r['qid'] not in pairwise_difference[r['model']]:
            pairwise_difference[r['model']][r['qid']] = {}
        if r['prev'] not in pairwise_difference[r['model']][r['qid']]:
            pairwise_difference[r['model']][r['qid']][r['prev']] = []
        pairwise_difference[r['model']][r['qid']][r['prev']].append(refDict[r['model']][r['qid']]-r['score'])

    bootstrapSamples = []
    for s in df[~df['model'].isin(["BM25", "RM3"])]['model'].unique():
        '''
        pairwise_difference_temp = {}
        for u1 in pairwise_difference[s]:
            pairwise_difference_temp[u1] = {}
            for u2 in pairwise_difference[s]:
                print(s, u1, u2)
                print(pairwise_difference[s][u1])

                pairwise_difference_temp[u1][u2] = [x for s2 in pairwise_difference for x in pairwise_difference[s2][u1][u2] if s2 != s and u1.split("_")[0]==u2.split("_")[0]]
        
        pairwise_difference_temp = {u1:
                                        {u2:
                                             [x for s2 in pairwise_difference for x in pairwise_difference[s2][u1][u2] if s2!=s]
                                         for u2 in pairwise_difference[s]}
                                    for u1 in pairwise_difference[s]}
        '''


        for t in perms:
            pairwise_difference_temp = {}
            for u1 in perms[t][0]:
                pairwise_difference_temp[u1] = {}
                for u2 in perms[t][0]:
                    pairwise_difference_temp[u1][u2]=[x for s2 in pairwise_difference for x in pairwise_difference[s2][u1][u2] if s2 != s]

            newConvs = [random.sample(list(perms[t][0]), k=len(perms[t][0])) for b in range(B)]
            bootstrapSamples += [[s, t, k,
                                np.mean([refDict[s][nc[0]]] + [refDict[s][u2] + random.choice(pairwise_difference_temp[u2][u1]) for u1, u2 in zip(nc[:-1], nc[1:])])
                                ] for k, nc in enumerate(newConvs)]

        print(len(bootstrapSamples))
    bootstrapSamples = pd.DataFrame(bootstrapSamples, columns=['model', 'topic', 'k', 'score'])
    return bootstrapSamples

def deltaBootstrap(df, B=10000):
    diffDf = df[(~df['model'].isin(['BM25', 'RM3'])) & (df['perm'] == 0)].copy()

    diffDf['prev_score'] = diffDf.apply(lambda x: getPrevScore(x, diffDf), axis=1)

    diffDf['diffs'] = diffDf['score'] - diffDf['prev_score']

    bootstrapSamples = _boostsrapDiffs(diffDf, B)

    return bootstrapSamples


def _boostsrapDiffs(diffDf, B):
    diffDict = {}
    for s in diffDf['model'].unique():
        diffDict[s] = {}
        for t in diffDf['topic'].unique():
            diffDict[s][t] = diffDf[(diffDf['model'] == s) & (diffDf['topic'] == t)]['diffs'].values

    bootstrapSamples = {}
    for index, row in diffDf.iterrows():
        if row['model'] not in bootstrapSamples:
            bootstrapSamples[row['model']] = {}
        if row['topic'] not in bootstrapSamples[row['model']]:
            bootstrapSamples[row['model']][row['topic']] = {}

        bootstrapSamples[row['model']][row['topic']][row['utterance']] = bootstrapSampling(row, diffDict)

    bootstrapSamplesMeans = []
    for s in sorted(bootstrapSamples):
        for t in sorted(bootstrapSamples[s]):
            for i in range(B):
                bootstrapSamplesMeans.append(
                    [s, t, i, np.mean([random.choices(bootstrapSamples[s][t][u]) for u in bootstrapSamples[s][t]])])

    bootstrapSamplesMeans = pd.DataFrame(bootstrapSamplesMeans, columns=['model', 'topic', 'k', 'score'])

    return bootstrapSamplesMeans


def refBootstrap(df, perms, B=10000):
    ### UTILITY FUNCTIONS

    def getPreviousUtterance(x):
        return perms[x['topic']][x['perm']][max(x['order'] - 1, 0)]

    #######
    df['prev'] = df.apply(getPreviousUtterance, axis=1)
    df[['prev_topic', 'prev_utterance']] = df['prev'].str.split("_", 1, expand=True)



    means = df[['topic', 'model', 'qid', 'utterance', 'prev', 'score', 'prev_utterance']].groupby(
        ['model', 'topic', 'qid', 'utterance', 'prev', 'prev_utterance']).aggregate('mean').reset_index()

    d = {a:
             {x:
                 {z:k['score'].values[0] for z, k in y.drop('qid', 1).groupby('prev')}
              for x, y in b.drop('model', 1).groupby('qid')}
         for a, b in means[['model', 'qid', 'prev', 'score']].groupby('model')}


    #means[['topic', 'utterance']] = means['qid'].str.split("_", 1, expand=True)

    bootstrapSamples = []
    for s in sorted(means[~means['model'].isin(['BM25', 'RM3'])]['model'].unique()):
        for t in sorted(means['topic'].unique()):
            utts = list(perms[t][0])
            for b in range(B):
                sampled_utts = random.sample(utts, k=len(utts))
                bootstrapSamples.append([s, t, b, np.mean([d[s][u1][u2] if u1 in d[s] and u2 in d[s][u1] else 0 for u1, u2 in zip(sampled_utts[:-1],sampled_utts[1:])])])

    bootstrapSamples = pd.DataFrame(bootstrapSamples, columns=['model', 'topic', 'k', 'score'])
    return bootstrapSamples

def bootstrapSampling(x, samples):

    samples = random.choices(samples[x['model']][x['topic']], k=len(samples[x['model']][x['topic']]))

    return [min(max(0, x['score']+s), 1) for s in samples]

def getPrevScore(x, data):
    return data[(data['topic'] == x['topic']) & (data['model'] == x['model']) & (data['order'] == max(0, x['order'] - 1))][
              'score'].values[0]

