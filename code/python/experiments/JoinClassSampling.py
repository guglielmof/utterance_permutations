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


class JoinClassSampling(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):
        conversations = ['31', '32', '33', '34', '37', '40', '49', '50', '54', '56', '58', '59', '61', '67', '68', '69',
                         '75', '77', '78', '79']

        ds = []
        for c in conversations:
            ds.append(pd.read_csv(f"../../data/measures/{c}.csv"))

        measures = pd.concat(ds)

        measures[['topic', 'utterance']] = measures['qid'].str.split("_", 1, expand=True)

        avgs = measures[~measures['name'].isin(['BM25', 'RM3'])]\
                        [['topic', 'perm', 'name', 'nDCG@3']]\
                        .groupby(['topic', 'name', 'perm'])\
                        .aggregate(['mean'])\
                        .droplevel(1, axis=1)\
                        .reset_index()

        print(avgs)

        sns.boxplot(data=avgs, x='topic', y='nDCG@3', hue='name', width=len(avgs['name'].unique())*0.15)
        systems = list(avgs['name'].unique())
        for e, nm in enumerate(systems):

            offset = (e-(len(systems)//2)) * 0.15


            x = np.arange(len(avgs['topic'].unique())) + offset
            sns.scatterplot(x=x,
                            y=avgs[(avgs["name"] == nm) & (avgs['perm']==0)]["nDCG@3"],
                            color='y',
                            edgecolor='k',
                            marker="d", zorder=10, s=100)


        df = pd.read_csv("../../data/measures/permutated_conversations_resolved_100.csv")
        df[['topic', 'utterance']] = df['qid'].str.split("_", 1, expand=True)

        avgs2 = df[~df['name'].isin(['BM25', 'RM3'])]\
                        [['topic', 'perm', 'name', 'nDCG@3']]\
                        .groupby(['topic', 'name', 'perm'])\
                        .aggregate(['mean'])\
                        .droplevel(1, axis=1)\
                        .reset_index()


        avgs2.columns = ['topic', 'name', 'perm', 'score']

        print(avgs[avgs['perm']==0])
        print(avgs2[avgs2['perm']==0])

        joined = pd.merge(avgs[avgs['perm']==0], avgs2[avgs2['perm']==0], how='inner', on=['name', 'topic', 'perm'])

        joined['diff'] = joined['nDCG@3'] - joined['score']

        print(joined)