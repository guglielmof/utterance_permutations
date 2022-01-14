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

from collections import Counter

import seaborn as sns
import random

class BootstrappingDifferences(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):
        df = pd.read_csv("../../data/measures/permutated_conversations_resolved_100.csv")
        df[['topic', 'utterance']] = df['qid'].str.split("_", 1, expand=True)

        df = df.drop(['Unnamed: 0', 'Unnamed: 0.1',
                      'AP', 'P@1', 'P@3', 'RR@10', 'nDCG@10',
                      ], axis=1)

        diffDf = df[(~df['name'].isin(['BM25', 'RM3'])) & (df['perm'] == 0)].copy()
        #diffDf = df[(df['perm'] == 0)].copy()

        diffDf['prev_score'] = diffDf.apply(lambda x: getPrevScore(x, diffDf), axis=1)

        diffDf['diffs'] = diffDf['nDCG@3']-diffDf['prev_score']

        #sns.histplot(data=diffDf, x='diffs', hue='name')


        diffDict = {}
        for s in diffDf['name'].unique():
            diffDict[s] = {}
            for t in diffDf['topic'].unique():
                diffDict[s][t] = diffDf[(diffDf['name']==s) & (diffDf['topic']==t)]['diffs'].values

        bootstrapSamples = {}
        for index, row in diffDf.iterrows():
            if row['name'] not in bootstrapSamples:
                bootstrapSamples[row['name']] = {}
            if row['topic'] not in bootstrapSamples[row['name']]:
                bootstrapSamples[row['name']][row['topic']] = {}
                    #if row['utterance'] not in bootstrapSamples[row['name']][row['topic']]:

            bootstrapSamples[row['name']][row['topic']][row['utterance']] = bootstrapSampling(row, diffDict)


        bootstrapSamplesMeans = []
        for s in sorted(bootstrapSamples):
            for t in sorted(bootstrapSamples[s]):
                for i in range(1001):
                    bootstrapSamplesMeans.append([s, t, i, np.mean([random.choices(bootstrapSamples[s][t][u]) for u in bootstrapSamples[s][t]])])

        bootstrapSamplesMeans = pd.DataFrame(bootstrapSamplesMeans, columns=['name', 'topic', 'k', 'score'])
        plt.figure(figsize=(20, 11))

        sns.boxplot(data=bootstrapSamplesMeans, x='topic', y='score', hue='name', width=len(bootstrapSamplesMeans['name'].unique())*0.15)


        aggregationBootstrap = bootstrapSamplesMeans[['name', 'topic', 'score']].groupby(['name', 'topic']).aggregate(['mean', 'std']).reset_index()


        aggregationBootstrap.columns = ['method', 'topic', 'mean_bootst', 'std_bootst']

        print(aggregationBootstrap)

        aggregationStandard  = df[~df['name'].isin(['BM25', 'RM3'])][['name', 'topic', 'perm', 'nDCG@3']].groupby(['name', 'topic', 'perm']).mean().reset_index()
        aggregationStandard  = aggregationStandard[['name', 'topic', 'nDCG@3']].groupby(['name', 'topic']).aggregate(['mean', 'std']).reset_index()

        aggregationStandard.columns = ['method', 'topic', 'mean_ndcg@3', 'std_ndcg@3']

        print(aggregationStandard)

        df_concat = pd.merge(aggregationBootstrap, aggregationStandard, how='inner') #pd.concat([aggregationBootstrap, aggregationStandard], axis=1)

        plt.figure(figsize=(20, 11))
        sns.scatterplot(data=df_concat, x='mean_ndcg@3', y='mean_bootst', style='method', hue='method')


        RSS = np.sum((df_concat['mean_ndcg@3'].values - df_concat['mean_bootst'].values)**2)
        TSS = np.sum(df_concat['mean_ndcg@3'].values - np.mean(df_concat['mean_ndcg@3'].values)**2)
        R2 = 1 - RSS/TSS
        plt.title(f"r2: {R2}")


        plt.figure(figsize=(20, 11))
        sns.scatterplot(data=df_concat, y='std_bootst', x='std_ndcg@3', style='method', hue='method')

        RSS = np.sum((df_concat['std_ndcg@3'].values - df_concat['std_bootst'].values)**2)
        TSS = np.sum(df_concat['std_ndcg@3'].values - np.mean(df_concat['std_ndcg@3'].values)**2)
        R2 = 1 - RSS/TSS
        plt.title(f"r2: {R2}")


        zero = df[(~df['name'].isin(['BM25', 'RM3']))& (df['perm']==0)][['name', 'topic', 'nDCG@3']].groupby(['name', 'topic']).mean().reset_index()

        zero.columns = ['method', 'topic', 'mean_ndcg@3_zero']

        wozero = df[(~df['name'].isin(['BM25', 'RM3']))& (df['perm']!=0)][['name', 'topic', 'perm', 'nDCG@3']].groupby(['name', 'perm', 'topic']).mean().reset_index()
        wozero = wozero[['name', 'topic', 'nDCG@3']].groupby(['name', 'topic']).aggregate(['mean']).reset_index()
        wozero.columns = ['method', 'topic', 'mean_ndcg@3_wozero']

        df_concat = pd.merge(zero, wozero, how='inner') #pd.concat([aggregationBootstrap, aggregationStandard], axis=1)

        plt.figure(figsize=(20, 11))
        sns.scatterplot(data=df_concat, x='mean_ndcg@3_zero', y='mean_ndcg@3_wozero', style='method', hue='method')
        plt.title(f"r2: {r2_score(zero['mean_ndcg@3_zero'].values, wozero['mean_ndcg@3_wozero'])}")
        RSS = np.sum((df_concat['mean_ndcg@3_zero'].values - df_concat['mean_ndcg@3_wozero'].values)**2)
        TSS = np.sum(df_concat['mean_ndcg@3_zero'].values - np.mean(df_concat['mean_ndcg@3_zero'].values)**2)
        R2 = 1 - RSS/TSS
        plt.title(f"r2: {R2}")


def bootstrapSampling(x, samples):

    samples = random.choices(samples[x['name']][x['topic']], k=len(samples[x['name']][x['topic']]))

    #return [[x['name'], x['topic'], x['utterance'], x['nDCG@3']+s] for s in samples]

    return [min(max(0, x['nDCG@3']+s), 1) for s in samples]

def getPrevScore(x, data):
    return data[(data['topic'] == x['topic']) & (data['name'] == x['name']) & (data['order'] == max(0, x['order'] - 1))][
              'nDCG@3'].values[0]
