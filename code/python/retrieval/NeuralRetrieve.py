import os
import pandas as pd
from pyterrier.measures import *
import pyterrier as pt
import sys

from utils import chunk_based_on_number


from functools import partial

from CAsTutils import loadIndex, loadQueries, loadQrels

import numpy as np

os.environ['JAVA_HOME'] = '../../../../SOFTWARE/jdk-11.0.11'

'''
import neuralcoref
nlp = spacy.load("en_core_web_sm")

neuralcoref.add_to_pipe(nlp)
doc_with_coref = nlp(v)
print(doc_with_coref)
new_conv_coref = doc_with_coref._.coref_resolved
new_doc_coref = nlp(new_conv_coref)
new_utt_list = list(new_doc_coref.sents)

print(new_utt_list)
'''


if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


queries = loadQueries("original")
qrels = loadQrels()

queries_resolved = loadQueries("resolved")

from allennlp_models import pretrained
from allennlp.predictors.predictor import Predictor
#predictor = Predictor.from_path(
#            "https://storage.googleapis.com/allennlp-public-models/coref-model-2018.02.05-allennlpv1.0.tar.gz")
predictor = pretrained.load_predictor("coref-spanbert")
#"https://storage.googleapis.com/allennlp-public-models/coref-model-2018.02.05-allennlpv1.0.tar.gz"

import spacy

#nlp = spacy.load("en_core_web_sm")



n_perms = 5
t2perm = {}
with open("../../../data/class_perms.txt", "r") as F:
    for l in F.readlines():
        topic, permutations = l.strip().split("\t")
        permutations = eval(permutations)
        t2perm[topic] = [sorted(permutations[0])] + permutations[1:n_perms]


def convert_topic_permutations(ds, t2perm):
    permutations = t2perm[ds['topic'].values[0]]
    order = {i:e for e, i in enumerate(sorted(permutations[0]))}
    for e, p in enumerate(permutations):
        newOrder = [order[i] for i in p]
        queries = ds.iloc[newOrder]

        print(queries)

        v = " | ".join(queries['query'].values)
        doc = predictor.coref_resolved(v)
        new_utt_list = doc.split(" | ")

        for e2, u in enumerate(new_utt_list):
            if u!=queries_resolved[queries_resolved['topic']==ds['topic'].values[0]]['query'].values[newOrder[e2]]:
                print(f"{u} - {queries_resolved[queries_resolved['topic']==ds['topic'].values[0]]['query'].values[newOrder[e2]]}")

queries.groupby(['topic']).apply(convert_topic_permutations, t2perm)
