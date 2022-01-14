import os
import pandas as pd
from pyterrier.measures import *
import pyterrier as pt
import sys

from utils import chunk_based_on_number


from functools import partial

from CAsTutils import loadIndex, loadQueries, loadQrels

import numpy as np
import os

os.environ['JAVA_HOME'] = '../../../../SOFTWARE/jdk-11.0.11'

if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


queries = loadQueries("resolved")
queries_original = loadQueries("original")
qrels = loadQrels()

from MyPyterrierPipelines import *


chunks = chunk_based_on_number(list(queries['topic'].unique()), 20)

def ComputeScores(queries, index, qrels, permutations, qtype='resolved'):



    std_models = [bm25(index), rm3(index), dirichletLM(index)]
    #std_models = [bm25(index)]

    seq_models = [allennlp_simple(index), first_query(index), context_query(index), linear_prev(index), rm3_prev(index), rm3_seq(index)]
    #seq_models = []

    scores = []

    order = {i: e for e, i in enumerate(permutations[0])}

    for e, p in enumerate(permutations):

        newOrder = [order[i] for i in p]
        queries2 = queries.iloc[newOrder]

        if e==0:
            models = std_models+seq_models
        else:
            models = seq_models
        for rm in models:
            result = rm.getConvScores(queries2, qrels)
            result['perm'] = [e]*len(p)
            result['qtype'] = [qtype] * len(p)
            scores.append(result)
            print(e, rm.name, np.mean(result['nDCG@3'].values))
            sys.stdout.flush()
    output = pd.concat(scores)
    return output


n_threads = 20
n_perms = 48
t2perm = {}
index = loadIndex()
with open("../../../data/class_perms_2.txt", "r") as F:
    for l in F.readlines():
        topic, permutations = l.strip().split("\t")
        permutations = eval(permutations)
        t2perm[topic] = [sorted(permutations[0])] + permutations[1:n_perms]


queries = queries[queries['topic']==sys.argv[1]]
queries_original = queries_original[queries_original['topic']==sys.argv[1]]

#result_resolved = ComputeScores(queries, index, qrels, t2perm[sys.argv[1]], qtype='resolved')
result_original = ComputeScores(queries_original, index, qrels, t2perm[sys.argv[1]], qtype='original')

#result = pd.concat([result_original, result_resolved])
result = result_original
result.to_csv(f"../../../data/measures/{sys.argv[1]}.csv")