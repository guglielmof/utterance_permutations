from . import AbstractExperiment
import experimentalCollections as tc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import json
import pandas as pd
import convLib

import matplotlib.colors as colors

class SimpleDebug(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):



        collection =  getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors, conv=True)
        with open("../../data/conv_annotations.json", "r") as fp:
            myAnnotations = json.load(fp)


        offset = 1
        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):
            new_cmap = truncate_colormap(cm.Greys, 0.2, 1.)
            c2t = {collection.conv2utt_ts[c][et]: collection.conv_ts[ec+offset]['turn'][et]['raw_utterance'] for et in range(len(collection.conv2utt_ts[c]))}

            ######################### g1 ######################À
            g1 = convLib.getGraph(collection.conv2utt_ts[c], collection.qrels_ts)
            print(g1)
            G1 = nx.from_pandas_edgelist(g1, source='Source', target='Target', edge_attr='weight')
            pos = nx.spring_layout(G1)
            edges, weights = zip(*nx.get_edge_attributes(G1, 'weight').items())
            nds = G1.nodes
            cnds = ['#1f78b4' if n in collection.conv2utt_ts[c] else 'red' for n in nds]
            snds = [30 if n in collection.conv2utt_ts[c] else 15 for n in nds]
            nx.draw_networkx(G1, pos, with_labels=True, labels=c2t, edge_color=weights, edge_cmap=new_cmap, node_color=cnds, node_size=snds)

            plt.show()
            plt.figure()

            ######################### g2 ######################À
            g2 = convLib.getCompressedGraph(collection.conv2utt_ts[c], collection.qrels_ts)

            G2 = nx.from_pandas_edgelist(g2, source='Source', target='Target', edge_attr='weight')


            edge_labels = dict([((n1, n2), d['weight'])
                                for n1, n2, d in G2.edges(data=True)])
            edges, weights = zip(*nx.get_edge_attributes(G2, 'weight').items())
            nx.draw_networkx(G2, pos, with_labels=True, labels=c2t, edge_color=weights, edge_cmap=new_cmap, node_size=30)
            nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels)
            plt.show()

            ######################### g3 ######################À
            myg = []
            for mn in myAnnotations:
                if str(mn['number'])==c:
                    for t, inlinks in enumerate(mn['links']):
                        for s, inlink in enumerate(inlinks):
                            if inlink == 1:
                                myg.append([collection.conv2utt_ts[c][s], collection.conv2utt_ts[c][t], 'Directed', 1])
                    break

            myg = pd.DataFrame(myg, columns=['Source', 'Target', 'Type', 'weight'])
            print(myg)
            myG = nx.from_pandas_edgelist(myg, source='Source', target='Target', edge_attr='weight', create_using=nx.DiGraph)

            plt.figure()
            nx.draw_networkx(myG, pos, with_labels=True, labels=c2t, node_size=30)
            #nx.draw_networkx_edge_labels(myG, pos, edge_labels=edge_labels)
            plt.show()
            break



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


