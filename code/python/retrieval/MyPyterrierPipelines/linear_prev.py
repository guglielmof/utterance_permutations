from .AbstractPipeline import AbstractPipeline
import pyterrier as pt




class linear_prev(AbstractPipeline):

    def __init__(self, index, **kwargs):
        self.name = "linear_prev"
        secondstage = (kwargs['secondstage'] if 'secondstage' in kwargs else pt.BatchRetrieve(index, wmodel="BM25"))
        weight = (kwargs['weight'] if 'weigth' in kwargs else 0.6)

        # pt.apply.query(lambda row: prevQry(row)) is a rewriter that rewrites the current query with the previous one

        # the next rewriter combines linearly the current query with the previous one linearly
        prev_rewriter = pt.apply.query(lambda row: self.getPrevQuery(row)) >> pt.rewrite.linear(weight, 1-weight)
        self.pipeline = prev_rewriter >> secondstage



    def getConvScores(self, conv, qrels):

        self.prevQueryDict = self.computePrevQueryDict(conv)
        return self.baseExperiment(conv, qrels)