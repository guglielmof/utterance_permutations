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

import experimentalCollections as EC

class AnnotateDocuments(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        collection = getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors, conv=True)
        msMarcoDocs = EC.readMSMARCO()
        carDocs = {}

        offset = 1
        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):
            new_cmap = utils.truncate_colormap(cm.Greys, 0.2, 1.)

            g = convLib.getGraph(collection.conv2utt_ts[c], collection.qrels_ts)

            rDocs = {d:msMarcoDocs[d] if d in msMarcoDocs else carDocs[d] for d in set(g['Target']) if d in msMarcoDocs or d in carDocs}


            annotatedDocs = {d:Annotator.annotate(rDocs[d]) for d in rDocs}
            #print(annotatedDocs)

            '''
            manualQueries = {"39_1": "What is throat cancer?",
                            "39_2": "Is THROAT CANCER treatable?",
                            "39_3": "Tell me about lung cancer.",
                            "39_4": "What are LUNG CANCER symptoms? ",
                            "39_5": "Can LUNG CANCER spread to the throat?",
                            "39_6": "What causes throat cancer?",
                            "39_7": "What is the first sign of THROAT CANCER?",
                            "39_8": "Is THROAT CANCER the same as esophageal cancer?",
                            "39_9": "What's the difference in ESOPHAGEAL CANCER AND THROAT CANCER symptoms?"}
            '''

            manualQueries = {"32_1" : "What are the different types of sharks?",
                            "32_2" : "Are sharks endangered?  If so, which species?",
                            "32_3" : "Tell me more about tiger sharks.",
                            "32_4" : "What is the largest SHARK ever to have lived on Earth?",
                            "32_5" : "What's the biggest SHARK ever caught?",
                            "32_6" : "What about for great whites SHARKS?",
                            "32_7" : "Tell me about makos SHARKS.",
                            "32_8" : "What are the MAKO SHARKS adaptations?",
                            "32_9" : "Where do MAKO SHARKS live?",
                            "32_10" : "What do MAKO SHARKS eat?",
                            "32_11" : "How do MAKO SHARKS compare with tiger SHARKS for being dangerous?"}
            annotatedManualQueries = {q:Annotator.annotate(manualQueries[q]) for q in manualQueries}

            g2 = convLib.getConceptBasedGraph(annotatedManualQueries, annotatedDocs, collection.qrels_ts)

            g2 = g2[(g2['Source'].isin(g2['Target'])) & (g2['Target'].isin(g2['Source']))]
            print(list(g2[['weight']]))
            G2 = nx.from_pandas_edgelist(g2[['Source', 'Target', 'Type', 'weight']], source='Source', target='Target', edge_attr='weight',create_using=nx.DiGraph)
            pos = nx.spring_layout(G2)
            edges, weights = zip(*nx.get_edge_attributes(G2, 'weight').items())
            nds = G2.nodes
            cnds = ['#1f78b4' if n in collection.conv2utt_ts[c] else 'red' for n in nds]
            snds = [30 if n in collection.conv2utt_ts[c] else 15 for n in nds]
            fig = plt.figure(figsize=(40, 22))
            nx.draw_networkx(G2, pos, with_labels=True, labels=manualQueries, edge_color=weights, edge_cmap=new_cmap, node_color=cnds, node_size=snds)
            plt.savefig("fig.png")
            break
