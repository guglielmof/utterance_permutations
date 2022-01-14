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

class AnalyzePermutationScoresDistribution_2(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        df = pd.read_csv("../../data/measures/permutated_conversations_resolved_100.csv")
        df[['topic', 'utterance']] = df['qid'].str.split("_", 1, expand=True)

        df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)


        perms = {t:
                     {p: df[(df['perm']==p) & (df['topic'] == t) & (df['name']=='RM3_prev')].sort_values('order', ascending=True)['qid'].values
                      for p in df['perm'].unique()}
                 for t in df['topic'].unique()}

        #print(perms)

        def getPreviousUtterance(x):

            return perms[x['topic']][x['perm']][max(x['order']-1, 0)]
            #return(df[(df['perm']==x['perm']) & (df['topic']==x['topic']) & (df['order']==max(x['order']-1, 0))]['qid'].values[0])


        df['prev'] = df.apply(getPreviousUtterance, axis=1)
        df[['prev_topic', 'prev_utterance']] = df['prev'].str.split("_", 1, expand=True)

        print(df)
        '''
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(df[df['name']=='RM3_prev'][['qid', 'perm', 'order', 'topic', 'utterance', 'prev']].head(100))
        '''

        means = df[['topic', 'name', 'qid', 'utterance', 'prev', 'nDCG@3', 'prev_utterance']].groupby(['name', 'qid', 'utterance', 'prev', 'prev_utterance']).aggregate('mean').reset_index()

        means[['topic', 'utterance']] = means['qid'].str.split("_", 1, expand=True)


        means['utterance'] = pd.to_numeric(means['utterance'])
        means['prev_utterance'] = pd.to_numeric(means['prev_utterance'])



        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(means[(means['name']=='RM3_prev')].head(100))

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            sns.heatmap(d, **kwargs)


        print(means['name'].unique())

        for n in means['name'].unique():
            print(n)
            #plt.figure()
            means_filt = means[means['name']==n]
            fg = sns.FacetGrid(means_filt, col='topic', col_wrap=5, sharex=False, sharey=False)
            fg.map_dataframe(draw_heatmap, 'utterance', 'prev_utterance', 'nDCG@3', cbar=False)
