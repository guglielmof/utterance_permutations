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

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import flatten, padded_everygram_pipeline

from sklearn.preprocessing import normalize

from collections import Counter
from nltk.stem.porter import *

class LMGraphs(AbstractExperiment):

    def __init__(self, **kwargs):
        nltk.download("stopwords")
        nltk.download("punkt")
        super().__init__(**kwargs)


    def run_experiment(self):
        stime = time.time()
        passagesText = EC.readReducedCollection()

        stop_words = set(stopwords.words('english'))
        '''
        docsTokenized = [word_tokenize(s) for _, s in msMarcoDocs.items()]+[word_tokenize(s) for _, s in carDocs.items()]
        word_tokens = list(flatten(t for t in docsTokenized))
        word_tokens = set([w for w in word_tokens if w not in stop_words and w.isalnum()])
        vocab = list(word_tokens)
        print(len(vocab))
        '''


        #selected_runs = ['input.mpi-d5_cqw']

        selected_runs = None
        #selected_runs = ['input.ilps-bert-feat1','input.pg2bert', 'input.pgbert', 'input.h2oloo_RUN2', 'input.CFDA_CLIP_RUN7']
        simMtrcs = []
        collection = getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors, conv=True, selected_runs=selected_runs)
        collection.runs = {r: rl for r, rl in collection.runs.items() if r not in collection.manual_runs}
        collection.systems = list(collection.runs.keys())
        collection.evalRuns('map')


        dfMeasure = []
        for r in collection.measure:
            for q in collection.measure[r]:
                dfMeasure.append([r, q.split("_")[0], q, collection.measure[r][q]])

        dfMeasure = pd.DataFrame(dfMeasure, columns=["system", "conv", "query", "measure"])

        offset = 1
        ndocs = 10



        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):
            cMeasures = dfMeasure[dfMeasure["conv"]==c]
            cMeasures = cMeasures[["system", "measure"]]
            cMeasures = cMeasures.groupby("system").aggregate("mean").reset_index()
            cMeasures = cMeasures.sort_values("measure", ascending=False)["system"]
            selected_runs = list(cMeasures)[:5]
            print(selected_runs)
            lms = {}


            for q in collection.conv2utt_ts[c]:
                # Get the set of all relevant documents in a single document
                rdocs= " ".join([passagesText[d].lower()
                            for d, s in collection.qrels_ts[q].items() if s>0 if d in passagesText])

                # Compute the language model
                lms[q] = computeLM(rdocs, stop_words)

            simMatrix = getLMSimilarityMatrix(lms)

            simMtrcs.append(simMatrix)



            for sr in selected_runs:
                rankedLists = collection.runs[sr]
                lms = {}
                for e, q in enumerate(collection.conv2utt_ts[c]):
                    rankedList = [d for d, _ in
                                    sorted([(d, s) for d, s in rankedLists[q].items() if d in passagesText], key=lambda x: x[1])[:ndocs]]
                    sdocs = " ".join([passagesText[d] for d in rankedList])

                    lms[q] = computeLM(sdocs, stop_words)

                simMatrix = getLMSimilarityMatrix(lms)
                simMtrcs.append(simMatrix)


            fig, axs = plt.subplots(2, 3, figsize=(40, 22))
            for e, m in enumerate(simMtrcs):
                ax = axs[int(e / 3), e % 3]

                ax.set_title("relevant" if e==0 else selected_runs[e-1], fontsize=24)
                ax.imshow(m, cmap='hot', interpolation='nearest')
                ax.set_xticklabels(np.arange(len(collection.conv2utt_ts[c])), fontsize=18)
                ax.set_yticklabels(np.arange(len(collection.conv2utt_ts[c])), fontsize=18)


            plt.savefig("forse.png")
            break
        print(f"{time.time() - stime:.2f}")


def computeLM(doc, stop_words):
    #stemmer = lambda x: x
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in word_tokenize(doc) if w not in stop_words and w.isalnum()]
    lm = Counter(words)
    lm = {w: lm[w]/len(words) for w in lm}
    return lm

def getLMSimilarityMatrix(lms):

    # Get the entire vocabulary for all queries
    vocab = set().union(*[set(lms[q].keys()) for q in lms])

    # associate to each word an unique integer (position)
    w2p = {w: e for e, w in enumerate(vocab)}

    # convert the language models in a matrix
    lmMatrix = np.zeros((len(lms), len(vocab)))
    for e, q in enumerate(lms):
        for w in lms[q]:
            lmMatrix[e, w2p[w]] = lms[q][w]

    # normalize the matrix with l2 to easily compute the cosine similarity
    lmMatrix = normalize(lmMatrix)
    simMatrix = np.matmul(lmMatrix, lmMatrix.T)
    for i in range(len(lms)):
        simMatrix[i, i] = 0
    return simMatrix