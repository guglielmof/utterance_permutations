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

class TrimCollection(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        stime = time.time()
        msMarcoDocs = EC.readMSMARCO()
        carDocs = EC.readCAR()

        collection = getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors, conv=True)

        interestingDocs = set()

        for q in collection.qrels_ts:
            for d, _ in collection.qrels_ts[q].items():
                if d in msMarcoDocs or d in carDocs:
                    interestingDocs.add(d)

        for r in collection.runs:
            for q in collection.runs[r]:
                for d, _ in collection.runs[r][q].items():
                    if d in msMarcoDocs or d in carDocs:
                        interestingDocs.add(d)

        with open("../../data/processed_collections/ds_reduced.tsv", "w") as F:
            for d in interestingDocs:
                F.write(f"{d}\t{msMarcoDocs[d] if d in msMarcoDocs else carDocs[d]}")

        print(len(interestingDocs))