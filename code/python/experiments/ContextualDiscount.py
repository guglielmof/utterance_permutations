from . import AbstractExperiment

import matplotlib.pyplot as plt
import pandas as pd
import convLib
from convLib import contexts, Annotator
import utils
import time
import numpy as np

from retrieval.Analyzers import StandardAnalyzer

import experimentalCollections as EC


class ContextualDiscount(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):
        stime = time.time()

        analyzer = StandardAnalyzer()

        #import the collection
        collection = getattr(EC, self.collectionId)(logger=self.logger). \
            importCollection(nThreads=self.processors, conv=True)

        #import the corpus
        corpus = EC.readReducedCollection()

        #remove from the collection the manual runs
        collection.runs = {r: rl for r, rl in collection.runs.items() if r not in collection.manual_runs}
        collection.systems = list(collection.runs.keys())



        #evaluate the runs
        collection.evalRuns('map')


        #convert the measures to a data frame, to ease selection
        dfMeasure = []
        for r in collection.measure:
            for q in collection.measure[r]:
                dfMeasure.append([r, q.split("_")[0], q, collection.measure[r][q]])

        dfMeasure = pd.DataFrame(dfMeasure, columns=["system", "conv", "query", "measure"])


        offset = 1
        ndocs = 10

        for ec, c in enumerate(list(collection.conv2utt_ts.keys())[offset:]):

            referenceContexts = []
            for q in collection.conv2utt_ts[c]:
                relevantDocuments = [d for d, s in collection.qrels_ts[q].items() if s>0]
                referenceContexts.append(contexts.LinguisticContext(relevantDocuments, corpus, analyzer))
            print(len(referenceContexts))
            #get the best systems for a specific conversation
            cMeasures = dfMeasure[dfMeasure["conv"]==c][["system", "measure"]].\
                            groupby("system").aggregate("mean").\
                            reset_index().\
                            sort_values("measure", ascending=False)["system"]


            selected_runs = list(cMeasures)[:5]+list(cMeasures)[-5:]
            for r in selected_runs:
                print(f"---------------{r}---------------")
                for eq, q in enumerate(collection.conv2utt_ts[c][:-1]):
                    topNdocs = [d for d, s in sorted([(d, s) for d, s in collection.runs[r][q].items()], key=lambda x: -x[1]) if d in corpus][:ndocs]
                    qrContext = contexts.LinguisticContext(topNdocs, corpus, analyzer)
                    similarityContexts = qrContext.computeContextsSimilarity(referenceContexts)
                    des = eq+1+np.argmax(similarityContexts[eq+1:])
                    if des!=eq+1:
                        print(f"{eq+1}: {collection.conv_ts_resolved[collection.conv2utt_ts[c][eq+1]]} ({similarityContexts[eq+1]:.4f}) - "
                              f"{des}: {collection.conv_ts_resolved[collection.conv2utt_ts[c][des]]}  ({similarityContexts[des]:.4f})")
                    else:
                        print(f"{eq+1}: {collection.conv_ts_resolved[collection.conv2utt_ts[c][eq+1]]} ({similarityContexts[eq + 1]:.4f}) - BEST")

                print("\n\n")




            break


        self.logger.info(f"EXPERIMENT TERMINATED. Done in {time.time()-stime:.2f} seconds.")


