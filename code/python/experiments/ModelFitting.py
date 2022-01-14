from . import AbstractExperiment
import experimentalCollections as tc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import json
import pandas as pd



class ModelFitting(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):
        measures = pd.read_csv(f"../../data/measures/full_valid_100.csv")

        sampled = measures[(measures['topic']==31)]
        print(sampled)
        sampled.groupby(['name', 'perm']).apply(buildMatrices)


def buildMatrices(ds):
    x = np.zeros((len(ds.index), len(ds.index)))
    idxs = list(np.argsort(ds['utterance'].values))
    x[[idxs[0]]+idxs[:-1],idxs] = ds['nDCG@3'].values

    print(x)

    refMtrx = np.zeros((len(ds.index), len(ds.index)))
    refIdx = list(np.arange(len(ds.index)-1))
    refMtrx[[0]+refIdx[:-1], refIdx] = 1

    order_sim = np.sum(x*refMtrx)/(len(ds.index)-1)
    #scores_sim =