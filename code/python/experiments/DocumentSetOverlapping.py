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

class DocumentSetOverlapping(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        stime = time.time()
        #selected_runs = ['input.mpi-d5_cqw']

        #selected_runs = None
        selected_runs = ['input.ilps-bert-feat1','input.pg2bert', 'input.pgbert', 'input.h2oloo_RUN2', 'input.CFDA_CLIP_RUN7']
        collection = getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors, conv=True, selected_runs=selected_runs)
        msMarcoDocs = EC.readMSMARCO()
        carDocs = EC.readCAR()
        print(f"{time.time()-stime:.2f}")


        offset = 1
        ndocs = 10


        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):
            iConvMt = np.zeros((len(collection.conv2utt_ts[c]), len(collection.conv2utt_ts[c])))
            for e1, q1 in enumerate(collection.conv2utt_ts[c]):
                r1 = [d for d in collection.qrels_ts[q1] if d in collection.qrels_ts[q1] and collection.qrels_ts[q1][d]>0]
                for e2, q2 in enumerate(collection.conv2utt_ts[c]):
                    if q1 != q2:
                        r2 = [d for d in collection.qrels_ts[q2] if
                              d in collection.qrels_ts[q2] and collection.qrels_ts[q2][d] > 0]
                        iConvMt[e1, e2] = len(set(r1).intersection(set(r2)))/np.sqrt(len(set(r1))*len(set(r2)))

            print("ideal order:")
            print(favouredOrder(iConvMt))
            print("\n\n")

            for n, rankedLists in collection.runs.items():
                convMt = np.zeros((len(collection.conv2utt_ts[c]), len(collection.conv2utt_ts[c])))
                for e1, q1 in enumerate(collection.conv2utt_ts[c]):
                    for e2, q2 in enumerate(collection.conv2utt_ts[c]):
                        if q1!=q2:
                            convMt[e1, e2] = len(set(rankedLists[q1]).intersection(set(rankedLists[q2])))/np.sqrt(len(set(rankedLists[q1]))*len(set(rankedLists[q2])))


                print(n)
                print(convMt)
                print(favouredOrder(convMt))
                print("\n\n")
            break

def favouredOrder(mt):
    h, w = mt.shape
    ord = [0]
    for e1 in range(h):
        sorting = list(np.argsort(-mt[ord[-1]]))
        print(sorting)
        for s in sorting:
            if s not in ord:
                ord.append(s)
                break

    return ord
