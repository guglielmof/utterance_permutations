from .AbstractPipeline import AbstractPipeline
import pyterrier as pt




class context_query(AbstractPipeline):

    def __init__(self, index, **kwargs):
        self.name = "context_query"
        retriever = (kwargs['secondstage'] if 'secondstage' in kwargs else pt.BatchRetrieve(index, wmodel="BM25"))



        # the next rewriter combines linearly the current query with the previous one linearly
        contextRewriter = pt.apply.query(lambda row: " ".join([self.getPrevQuery(row), self.first_query, row['query']]))
        rewriter = contextRewriter
        self.rewriter = rewriter
        self.pipeline = rewriter >> retriever



    def getConvScores(self, conv, qrels):

        self.prevQueryDict = self.computePrevQueryDict(conv)
        return self.baseExperiment(conv, qrels)