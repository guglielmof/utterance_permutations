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

import experimentalCollections as EC

class NonRelevantAnalysis(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        stime = time.time()
        #selected_runs = ['input.mpi-d5_cqw']
        selected_runs = None
        collection = getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors, conv=True, selected_runs=selected_runs)
        msMarcoDocs = EC.readMSMARCO()
        carDocs = EC.readCAR()
        print(f"{time.time()-stime:.2f}")


        offset = 0
        ndocs = 10


        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):
            for n, rankedLists in collection.runs.items():
                for q in collection.conv2utt_ts[c]:

                    #take the non relevant documents: are those relevant for any other query?
                    nrd = {d: [q for q in  collection.conv2utt_ts[c] if d in collection.qrels_ts[q] and collection.qrels_ts[q][d]>=1]
                           for d, s in rankedLists[q].items() if d not in collection.qrels_ts[q] or collection.qrels_ts[q][d]<2}

                    print(collection.conv_ts_resolved[q])
                    print(nrd)
                break
            break