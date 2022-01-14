from .AbstractPipeline import AbstractPipeline
import pyterrier as pt
import pandas as pd


class rm3_seq(AbstractPipeline):

    def __init__(self, index, **kwargs):

        self.name = "RM3_seq"

        rm3_params = {i: kwargs[i] for i in kwargs if i in ['fb_terms', 'fb_docs', 'fb_lambda']}

        self.rm3_rewriter = pt.rewrite.RM3(index, **rm3_params)
        self.pipeline = pt.BatchRetrieve(index, wmodel="BM25")

    def getConvScores(self, conv, qrels):

        # first step: retrieve the ranked list for the first query
        df = self.baseExperiment(conv.iloc[[0]], qrels)
        currRankedList = self.pipeline(conv.iloc[[0]])

        for idx, row in conv.iloc[1:].iterrows():
            currRankedList['qid'] = [row['qid']]*len(currRankedList.index)
            currRankedList['query'] = [row['query']]*len(currRankedList.index)

            newQueryString = self.rm3_rewriter.transform(currRankedList)['query'].values[0].replace("applypipeline:off ", "")

            rewrittenQry = pd.DataFrame([[row['qid'], newQueryString]], columns=["qid", "query"])

            df = df.append(self.baseExperiment(rewrittenQry, qrels))
            currRankedList = self.pipeline(rewrittenQry)

        return df