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

class ManualConvInspection(AbstractExperiment):

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
                string = ""
                for q in collection.conv2utt_ts[c]:
                    docs = [did for did, _ in sorted([(d, s) for d, s in rankedLists[q].items()
                                                      if ('MARCO' in d or 'CAR' in d) and
                                                      (d in collection.qrels_ts[q] and collection.qrels_ts[q][d]>0)],
                                                     key=lambda x: -x[1])]
                    docs = {d: {'content': msMarcoDocs[d] if d in msMarcoDocs else carDocs[d],
                                'overlapping': [len(set(collection.conv_ts_resolved[q2].split()).intersection(
                                    set((msMarcoDocs[d] if d in msMarcoDocs else carDocs[d]).split())))/len(set(collection.conv_ts_resolved[q2].split()))
                                    for q2 in collection.conv2utt_ts[c]]
                                } for d in docs[:ndocs]}
                    string += f"{q}: {collection.conv_ts_resolved[q]}"
                    for d in docs:
                        string += f"\n{d}:\n{docs[d]}\n\n"
                    string += "\n\n"

                with open(f"../../../data/TREC/TREC_28_2019_CAsT/queryAndDocsBySystem/{n}_{c}_{ndocs}.txt", "w") as F:
                    F.write(string)

