import pandas as pd
import numpy as np

def getGraph(conv, qrels):
    graph = []

    for q in conv:
        for d, s in qrels[q].items():
            if s>0:
                graph.append([q, d, 'Undirected', s])


    graph = pd.DataFrame(graph, columns=['Source', 'Target', 'Type', 'weight'])

    return graph


def getCompressedGraph(conv, qrels):

    graph = []
    q2d = {}
    for q in conv:
        q2d[q] = set()
        for d, s in qrels[q].items():
            if s>0:
                q2d[q].add(d)


    for e1, q1 in enumerate(conv[:-1]):
        for e2, q2 in enumerate(conv[e1+1:]):
            its = len(q2d[q1].intersection(q2d[q2]))
            if its>0:
                graph.append([q1, q2, 'Undirected', its])


    graph = pd.DataFrame(graph, columns=['Source', 'Target', 'Type', 'weight'])

    return graph


def getConceptBasedGraph(qAnns, dAnns, qrels, function=None):

    graph = []

    d2q = {d: {q: set(dAnns[d]).intersection(set(qAnns[q])) for q in qAnns if len(set(dAnns[d]).intersection(set(qAnns[q])))>0} for d in dAnns}


    for q in qAnns:
        for d, s in qrels[q].items():
            if s > 0:
                graph.append([q, d, 'Directed', 'solid', s])


    for d in d2q:
        for q in d2q[d]:
            if d not in qrels[q] or qrels[q][d]>=1:
                graph.append([d, q, 'Directed', 'dashed', len(dAnns[d])/np.sqrt(len(dAnns[d])*len(qAnns[q]))])

    graph = pd.DataFrame(graph, columns=['Source', 'Target', 'Type', 'style', 'weight'])

    return graph