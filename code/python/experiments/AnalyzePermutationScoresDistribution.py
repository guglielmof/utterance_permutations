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

from collections import Counter

import seaborn as sns

class AnalyzePermutationScoresDistribution(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        df = pd.read_csv("../../data/measures/permutated_conversations_resolved_100.csv")



        df[['topic', 'utterance']] = df['qid'].str.split("_", 1, expand=True)

        measMeans = df.groupby(['name', 'topic', 'qtype', 'perm']).aggregate("mean").reset_index()
        measMeans = measMeans[['name', 'topic', 'qtype', 'perm', 'nDCG@3']]
        diffFromOriginal = lambda x: x['nDCG@3'] - measMeans[(measMeans['perm']==0)
                                               & (measMeans['topic']==x['topic'])
                                               & (measMeans['qtype'] == x['qtype'])
                                               & (measMeans['name']==x['name'])]['nDCG@3'].values[0]

        measMeans['diff'] = measMeans.apply(diffFromOriginal, axis=1)

        print(measMeans[measMeans['perm']==0][['name','perm', 'nDCG@3']].groupby(["name"]).aggregate("mean").reset_index())

        plt.figure(figsize=(20, 11))

        sns.boxplot(data=measMeans, x='topic', y='nDCG@3', hue='name', width=len(measMeans['name'].unique())*0.15)

        colors = ['k', 'r', 'g', 'b', 'y']
        #print(measMeans[measMeans['topic']=='69'])
        systems = list(measMeans['name'].unique())
        for e, nm in enumerate(systems):

            offset = (e-(len(systems)//2)) * 0.15


            x = np.arange(len(measMeans['topic'].unique())) + offset
            sns.scatterplot(x=x,
                            y=measMeans[(measMeans["name"] == nm) & (measMeans['perm']==0)]["nDCG@3"],
                            color='y',
                            edgecolor='k',
                            marker="d", zorder=10, s=100)


        plt.savefig("../../data/figures/boxplot.pdf")
        plt.close()


        plt.figure()
        sns.histplot(data=measMeans[~measMeans['name'].isin(['BM25', 'RM3'])], x='diff')
        plt.axvline(0, c='r', linewidth=2, dashes=(3,3))

        plt.savefig("../../data/figures/histogram.pdf")
        plt.close()


        maxDiff = measMeans.loc[measMeans[~measMeans['name'].isin(['BM25', 'RM3'])].groupby(['name', 'topic'])['diff'].idxmax()]

        fig = plt.figure()


        print(maxDiff)
        print(maxDiff[['topic', 'diff']].groupby("topic").aggregate("mean").reset_index())
        print(maxDiff[["name", "nDCG@3"]].groupby("name").aggregate("mean").reset_index())

        sns.barplot(data=maxDiff, x='topic', y='diff')
        plt.savefig("../../data/figures/max_barplot.pdf")
        plt.close()


        grid = sns.FacetGrid(measMeans[(~measMeans['name'].isin(['BM25', 'RM3']))], col="topic", col_wrap=5)
        grid.map(sns.histplot, "diff", "name")
        for (row_val, col_val), ax in grid.axes_dict.items():
            ax.axvline(0, c='r', linewidth=2, dashes=(3,3))

        plt.savefig("../../data/figures/grid.pdf")
        plt.close()

        #print(measMeans[(measMeans['topic'] == '40')])
        #print(measMeans[(measMeans['topic'] == '33')])




        rdocs = {}
        lms = {}
        collection = getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors, conv=True)

        # import the corpus
        corpus = EC.readReducedCollection()
        analyzer = StandardAnalyzer()

        for u in (df['qid'].unique()):
            relevantDocuments = [d for d, s in collection.qrels_ts[u].items() if s > 0]
            lms[u] = contexts.LinguisticContext(relevantDocuments, corpus, analyzer)


        lmStats = []
        permutations = {}
        for n in df['name'].unique():
            for t in df['topic'].unique():
                tmpDf = df[(df['topic']==t) & (df['name']==n)][['topic', 'perm', 'qid', 'order']]
                pairs = {}
                permutations[t] = {}
                for p in tmpDf['perm'].unique():
                    ordered_qid = tmpDf[tmpDf['perm']==p].sort_values(by = 'order')['qid'].values
                    permutations[t][p] = ordered_qid
                    similarities = []
                    for u1, u2 in zip(ordered_qid[1:], ordered_qid[:-1]):
                        if (u1, u2) not in pairs:
                            sim = lms[u1].computeContextsSimilarity([lms[u2]])[0]
                            pairs[(u1, u2)] = sim

                        similarities.append(pairs[(u1, u2)])
                    lmStats.append([n, t, p, np.mean(similarities)])


        diffFromOriginalQw = lambda x: x['nDCG@3'] - df[(df['perm']==0)
                                               & (df['qid']==x['qid'])
                                               & (df['name']==x['name'])]['nDCG@3'].values[0]
        diffGetPrevQuery = lambda x: permutations[x['topic']][x['perm']][max(0, x['order']-1)]


        #df['diff'] = df.apply(diffFromOriginalQw, axis=1)
        #df['prev_qid'] = df.apply(diffGetPrevQuery, axis=1)
        for t in df['topic'].unique():
            for n in df[~df['name'].isin(["BM25", "RM3"])]['name'].unique():
                tmpDf = df[(df['name']==n) & (df['topic']==t)]
                tmpDf['diff'] = tmpDf.apply(diffFromOriginalQw, axis=1)
                tmpDf['prev_qid'] = tmpDf.apply(diffGetPrevQuery, axis=1)

                print(tmpDf.groupby(['qid','prev_qid']).aggregate('mean'))

                selected_perms = list(measMeans[(measMeans['name']==n) & (measMeans['topic']==t) & (measMeans['diff']>0)]['perm'].values)
                print(selected_perms)
                tmpDfSP = tmpDf[tmpDf['perm'].isin(selected_perms)]
                tmpDfSP = tmpDfSP.groupby(['qid', 'prev_qid']).aggregate('mean')
                print(tmpDfSP)



        lmStats = pd.DataFrame(lmStats, columns=['name', 'topic', 'perm', 'mean_sim'])
        lmStats['mean_meas'] = lmStats.apply(lambda x: measMeans[(measMeans['name']==x['name']) &
                                                (measMeans['topic']==x['topic']) &
                                                (measMeans['perm']==x['perm'])]['nDCG@3'].values[0], axis=1)

        lmStats['diff'] = lmStats.apply(lambda x: measMeans[(measMeans['name']==x['name']) &
                                                (measMeans['topic']==x['topic']) &
                                                (measMeans['perm']==x['perm'])]['diff'].values[0], axis=1)

        plt.figure()
        sns.scatterplot(data=lmStats[(~lmStats['name'].isin(['BM25', 'RM3']))], x='mean_sim', y='mean_meas', hue='name', style='name')
        plt.savefig("../../data/figures/correlation_measure.pdf")
        plt.close()

        plt.figure()
        grid = sns.FacetGrid(lmStats[(~lmStats['name'].isin(['BM25', 'RM3']))], col="topic", col_wrap=5, legend_out=False)
        grid.map(sns.scatterplot, "mean_sim", "mean_meas", "name")
        grid.axes[0].legend()
        plt.savefig("../../data/figures/grid_correlation_measure.pdf")
        plt.close()


        plt.figure()
        sns.scatterplot(data=lmStats[(~lmStats['name'].isin(['BM25', 'RM3']))], x='mean_sim', y='diff', hue='name', style='name')
        plt.savefig("../../data/figures/correlation_diff.pdf")
        plt.close()


        plt.figure()
        grid = sns.FacetGrid(lmStats[(~lmStats['name'].isin(['BM25', 'RM3']))], col="topic", col_wrap=5, legend_out=False)
        grid.map(sns.scatterplot, "mean_sim", "diff", "name")
        grid.axes[0].legend()
        plt.savefig("../../data/figures/grid_correlation_diff.pdf")
        plt.close()
