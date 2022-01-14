import pyterrier as pt
from pyterrier.measures import *
import pandas as pd

class AbstractPipeline:
    pipeline = None
    name = "Undefined"

    def getPipeline(self):
        return self.pipeline

    def getConvScores(self, conv, qrels):
        return self.baseExperiment(conv, qrels)

    def baseExperiment(self, conv, qrels):
        self.first_query = conv['query'].iloc[0]


        conv = conv[['qid', 'query']]
        df = pt.Experiment(
            [self.pipeline],
            conv, qrels,
            drop_unused=True,
            #eval_metrics=[AP, RR @ 10, nDCG @ 10, nDCG @ 3, P @ 1, P @ 3],
            eval_metrics=[nDCG @ 3],
            names=[self.name],
            perquery=True
        )
        df['measure'] = df['measure'].map(str)
        df = df.pivot_table(values='value', index=['name', 'qid'], columns='measure').reset_index()
        return df

    def getSequentialConvScores(self, conv, qrels):
        raise NotImplementedError


    def computePrevQueryDict(self, conv):
        prevQueryDict = list(conv['query'].values[:-1])
        prevQueryDict = [prevQueryDict[0]] + prevQueryDict
        prevQueryDict = {x: y for x, y in zip(conv['qid'].values, prevQueryDict)}
        return prevQueryDict

    def getPrevQuery(self, row):
        try:
            return self.prevQueryDict[row['qid']]
        except AttributeError:
            print("prevQry method can be used only by those methods that compute the previous query")