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

class HDAPermutationScores(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        df = pd.read_csv("../../data/measures/permutated_conversations_resolved_100.csv")
        df[['topic', 'utterance']] = df['qid'].str.split("_", 1, expand=True)

        df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)


        with open("../../data/conv_annotations.json", "r") as F:
            annotations = json.load(F)

        annotations = {str(i['number']):np.array(i['links']) for i in annotations}

        conv_struct = {}
        for idx, adjMatrix in annotations.items():

            roots = [e for e, i in enumerate(adjMatrix) if np.all(i == 0)]
            leaves = [e for e, i in enumerate(adjMatrix.T) if np.all(i == 0)]

            parents = {e: list(np.where(i == 1)[0]) for e, i in enumerate(adjMatrix)}
            children = {e: list(np.where(i == 1)[0]) for e, i in enumerate(adjMatrix.T)}
            conv_struct[idx] = {'root': roots,'leaves': leaves, 'children': children, 'parents': parents}

        hdaf = lambda x: HDA(x, conv_struct, direction="forward")
        hdab = lambda x: HDA(x, conv_struct, direction="backward")
        aggregation = df[["name", "topic", "perm", "nDCG@3", "utterance"]].groupby(["name", "topic", "perm"]).apply(hdaf).reset_index()
        aggregation = aggregation.rename({0: 'aggr'}, axis=1)
        aggregation.to_csv("../../data/measures/aggregated_permutations_hdaf.csv")
        plt.figure(figsize=(20, 11))

        sns.boxplot(data=aggregation, x='topic', y='aggr', hue='name', width=len(aggregation['name'].unique())*0.15)

        colors = ['k', 'r', 'g', 'b', 'y']

        systems = list(aggregation['name'].unique())
        for e, nm in enumerate(systems):

            offset = (e-(len(systems)//2)) * 0.15


            x = np.arange(len(aggregation['topic'].unique())) + offset
            sns.scatterplot(x=x,
                            y=aggregation[(aggregation["name"] == nm) & (aggregation['perm']==0)]["aggr"],
                            color='y',
                            edgecolor='k',
                            marker="d", zorder=10, s=100)



def HDA(data, annotations, direction="forward"):


    topic = data['topic'].unique()[0]
    conv_struct = annotations[topic]

    utterances = [int(i) for i in data['utterance'].unique()]
    scores = np.array(data['nDCG@3'])
    sortedScores = scores[np.argsort(utterances)]

    if direction=='forward':
        return np.mean(np.array(HDA_f(conv_struct, sortedScores))[conv_struct['leaves']])

    if direction=='backward':
        return np.mean(np.array(HDA_b(conv_struct, sortedScores))[conv_struct['root']])



def HDA_b(conv_struct, measures):
    scores = [np.NaN for i in range(len(measures))]

    stack = [r for r in conv_struct['root']]
    while len(stack) > 0:
        node = stack[0]
        if conv_struct['children'][node] == []:
            scores[node] = measures[node]
            stack = stack[1:]
        else:
            children = conv_struct['children'][node]
            computable = True
            for c in children:
                if np.isnan(scores[c]):
                    stack = [c] + stack
                    computable = False

            if computable:
                scores[node] = (measures[node]) + (1 - measures[node]) * (np.mean([scores[c] for c in children]))
                stack = stack[1:]

    return scores


def HDA_f(conv_struct, measures):
    scores = [np.NaN for i in range(len(measures))]

    queue = list(conv_struct['parents'].keys())

    while len(queue) > 0:
        node = queue[0]
        if conv_struct['parents'][node] == []:
            scores[node] = measures[node]
            queue = queue[1:]
        else:
            parents = conv_struct['parents'][node]
            computable = True
            for c in parents:
                if np.isnan(scores[c]):
                    computable = False
                    queue = queue[1:] + [node]

            if computable:
                scores[node] = measures[node] + (1 - measures[node]) * (np.mean([scores[c] for c in parents]))
                queue = queue[1:]

    return scores