from . import AbstractExperiment

import pandas as pd
import convLib
from convLib import contexts, Annotator
import utils
import time
import numpy as np

from retrieval.Analyzers import StandardAnalyzer

import experimentalCollections as EC
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import seaborn as sns


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


new_cmap = truncate_colormap(cm.Greys, 0.2, 1.)


class ContextualDiscountGraph(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):
        stime = time.time()

        analyzer = StandardAnalyzer()

        # import the collection
        collection = getattr(EC, self.collectionId)(logger=self.logger). \
            importCollection(nThreads=self.processors, conv=True)

        # import the corpus
        corpus = EC.readReducedCollection()

        # remove from the collection the manual runs
        collection.runs = {r: rl for r, rl in collection.runs.items() if r not in collection.manual_runs}
        collection.systems = list(collection.runs.keys())

        # evaluate the runs
        collection.evalRuns('map')

        # convert the measures to a data frame, to ease selection
        dfMeasure = []
        for r in collection.measure:
            for q in collection.measure[r]:
                dfMeasure.append([r, q.split("_")[0], q, collection.measure[r][q]])

        dfMeasure = pd.DataFrame(dfMeasure, columns=["system", "conv", "query", "measure"])

        offset = 1
        ndocs = 10

        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):

            utterances = collection.conv2utt_ts[c]

            referenceContexts = []
            for q in utterances:
                relevantDocuments = [d for d, s in collection.qrels_ts[q].items() if s > 0]
                referenceContexts.append(contexts.LinguisticContext(relevantDocuments, corpus, analyzer))

            refGraph = []
            refSimMatrix = np.zeros((len(utterances), len(utterances)))
            for e1, u1 in enumerate(utterances[:-1]):
                similarityContexts = referenceContexts[e1].computeContextsSimilarity(referenceContexts[e1 + 1:])
                refSimMatrix[e1, e1 + 1:] = similarityContexts
                refSimMatrix[e1 + 1:, e1] = similarityContexts
                for e2, u2 in enumerate(utterances[e1 + 1:]):
                    refGraph.append([u1, u2, 'Undirected', similarityContexts[e2]])

            refGraph = pd.DataFrame(refGraph, columns=['Source', 'Target', 'Type', 'Weight'])

            gObj = nx.from_pandas_edgelist(refGraph, source='Source', target='Target', edge_attr='Weight')
            pos = plotGraph(gObj, refSimMatrix, utterances, c, "reference")
            plotRoutes(utterances, refSimMatrix, c, "reference", pos=pos)

            # get the best systems for a specific conversation
            cMeasures = dfMeasure[dfMeasure["conv"] == c][["system", "measure"]]. \
                groupby("system").aggregate("mean"). \
                reset_index(). \
                sort_values("measure", ascending=False)["system"]

            selected_runs = list(cMeasures)[:5] + list(cMeasures)[-5:]
            print(selected_runs)
            for r in selected_runs:

                runGraph = []
                runSimMatrix = np.zeros((len(utterances), len(utterances)))
                for eu1, u1 in enumerate(utterances[:-1]):
                    topNdocs = [d for d, s in
                                sorted([(d, s) for d, s in collection.runs[r][u1].items()], key=lambda x: -x[1]) if
                                d in corpus][:ndocs]
                    qrContext = contexts.LinguisticContext(topNdocs, corpus, analyzer)
                    similarityContexts = qrContext.computeContextsSimilarity(referenceContexts)
                    for eu2, u2 in enumerate(utterances[eu1 + 1:]):
                        runGraph.append([u1, u2, 'Undirected', similarityContexts[eu2]])

                    runSimMatrix[eu1, eu1 + 1:] = similarityContexts[eu1 + 1:]
                    runSimMatrix[eu1 + 1:, eu1] = similarityContexts[eu1 + 1:]

                runGraph = pd.DataFrame(runGraph, columns=['Source', 'Target', 'Type', 'Weight'])
                rgObj = nx.from_pandas_edgelist(runGraph, source='Source', target='Target', edge_attr='Weight')

                plotGraph(rgObj, runSimMatrix, utterances, c, r, pos=pos)
                plotRoutes(utterances, runSimMatrix, c, r, pos=pos)

            break

        self.logger.info(f"EXPERIMENT TERMINATED. Done in {time.time() - stime:.2f} seconds.")


def plotGraph(G, mtx, utterances, conv, name, pos=None):
    if pos is None:
        pos = nx.spring_layout(G)
    edges, weights = zip(*nx.get_edge_attributes(G, 'Weight').items())
    plt.figure(figsize=(20, 11))
    nx.draw_networkx(G, pos, with_labels=True, edge_color=weights, edge_cmap=new_cmap)
    labels = {k: f"{l:.2f}" for k, l in nx.get_edge_attributes(G, 'Weight').items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(f"../../data/contextualDiscountGraphs/{conv}/{name}_graph.png")
    plt.close()
    fig, ax = plt.subplots(figsize=(20, 11))
    sns.heatmap(mtx, annot=True, fmt='.2f', vmin=0, vmax=1)

    # ax.set_xticks(range(len(utterances)))
    # ax.set_yticks(range(len(utterances)))
    ax.set_xticklabels(utterances)
    ax.set_yticklabels(utterances)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
    plt.savefig(f"../../data/contextualDiscountGraphs/{conv}/{name}_matrix.png")
    plt.close()
    return pos


def plotRoutes(utterances, mtx, conv, name, pos=None):

    plt.figure(figsize=(20, 11))
    routes = [utterances]

    bestRoute = [0]
    to_be_considered = np.arange(len(utterances))
    to_be_considered = [c for c in to_be_considered if c!=bestRoute[-1]]
    while len(to_be_considered)>0:
        bestRoute.append(to_be_considered[np.argmax(mtx[bestRoute[-1], to_be_considered])])
        to_be_considered = [c for c in to_be_considered if c != bestRoute[-1]]

    routes.append([utterances[i] for i in bestRoute])

    desiredRoutes = []
    for e, rm in enumerate(mtx[:-1]):
        desiredRoutes.append([utterances[e], utterances[e+1+np.argmax(rm[e+1:])]])

    routes += desiredRoutes

    colors = ['r', 'b']+['y' for _ in desiredRoutes]

    linewidths = [3, 2] + [1 for _ in desiredRoutes]


    G = nx.DiGraph()
    edges = []
    for r in routes:
        route_edges = [(r[n], r[n + 1]) for n in range(len(r) - 1)]
        G.add_nodes_from(r)
        G.add_edges_from(route_edges)
        edges.append(route_edges)
    if pos is None:
        pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos)

    for ctr, edgelist in enumerate(edges):
        nx.draw_networkx_edges(G,pos=pos,edgelist=edgelist,edge_color = colors[ctr], width=linewidths[ctr])
    plt.savefig(f"../../data/contextualDiscountGraphs/{conv}/{name}_coloredRoutes.png")
    plt.close()

    return pos