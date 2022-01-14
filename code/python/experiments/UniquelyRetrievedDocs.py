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
import seaborn as sns


import experimentalCollections as EC

class UniquelyRetrievedDocs(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        stime = time.time()
        #selected_runs = ['input.mpi-d5_cqw']

        selected_runs = None
        #selected_runs = ['input.ilps-bert-feat1','input.pg2bert', 'input.pgbert', 'input.h2oloo_RUN2', 'input.CFDA_CLIP_RUN7']
        collection = getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors, conv=True, selected_runs=selected_runs)

        offset = 0
        docsRetrieved = {}
        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):
            docsRetrieved[c] = {}
            for e1, q1 in enumerate(collection.conv2utt_ts[c]):
                docsRetrieved[c][q1] = set().union(*[list(collection.runs[r][q1].keys()) for r in collection.runs])


        uniqueDocsRetrieved = {}
        uniqueRelevantDocsRetrieved = {}
        uniqueNonRelevantDocsRetrieved = {}
        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):
            uniqueDocsRetrieved[c] = {}
            uniqueRelevantDocsRetrieved[c] = {}
            uniqueNonRelevantDocsRetrieved[c] = {}
            for r in collection.runs:
                uniqueDocsRetrieved[c][r] = {}
                uniqueRelevantDocsRetrieved[c][r] = {}
                uniqueNonRelevantDocsRetrieved[c][r] = {}
                for e1, q1 in enumerate(collection.conv2utt_ts[c]):
                    others = set().union(*[list(collection.runs[r1][q1].keys()) for r1 in collection.runs if r1!=r])
                    uniqueDocsRetrieved[c][r][q1] = set(collection.runs[r][q1].keys()).difference(others)
                    uniqueRelevantDocsRetrieved[c][r][q1] = set([d for d in uniqueDocsRetrieved[c][r][q1] if d in collection.qrels_ts[q1] and collection.qrels_ts[q1][d]>0])
                    uniqueNonRelevantDocsRetrieved[c][r][q1] = set([d for d in uniqueDocsRetrieved[c][r][q1] if d not in collection.qrels_ts[q1] or collection.qrels_ts[q1][d]==0])




        uniqueDocsRetrievedA = []
        uniqueRelevantDocsRetrievedA = []
        uniqueNonRelevantDocsRetrievedA = []


        for c in uniqueDocsRetrieved:
            for r in uniqueDocsRetrieved[c]:
                for q in uniqueDocsRetrieved[c][r]:
                    uniqueDocsRetrievedA.append([c, q, r, len(uniqueDocsRetrieved[c][r][q])])
                    uniqueRelevantDocsRetrievedA.append([c, q, r, len(uniqueRelevantDocsRetrieved[c][r][q])])
                    uniqueNonRelevantDocsRetrievedA.append([c, q, r, len(uniqueNonRelevantDocsRetrieved[c][r][q])])


        uniqueDocsRetrieved = pd.DataFrame(uniqueDocsRetrievedA, columns=['conv', 'query', 'system', 'nUniques'])
        uniqueRelevantDocsRetrieved = pd.DataFrame(uniqueRelevantDocsRetrievedA, columns=['conv', 'query', 'system', 'nUniques'])
        uniqueNonRelevantDocsRetrieved = pd.DataFrame(uniqueNonRelevantDocsRetrievedA, columns=['conv', 'query', 'system', 'nUniques'])



        print(uniqueDocsRetrieved[uniqueDocsRetrieved['nUniques']>0])
        print(uniqueRelevantDocsRetrieved[uniqueRelevantDocsRetrieved['nUniques']>0])
        print(uniqueNonRelevantDocsRetrieved[uniqueNonRelevantDocsRetrieved['nUniques']>0])

        plt.figure()
        sns.histplot(data=uniqueDocsRetrieved[uniqueDocsRetrieved['nUniques']>0], x='nUniques', stat='probability')
        plt.xlim([0, 200])
        plt.savefig("uniqueDocsByQuerySystem.png")