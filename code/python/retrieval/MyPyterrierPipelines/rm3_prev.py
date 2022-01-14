from .AbstractPipeline import AbstractPipeline
import pyterrier as pt


class rm3_prev(AbstractPipeline):

    def __init__(self, index, **kwargs):
        self.name = "RM3_prev"

        firststage = (kwargs['firststage'] if 'firststage' in kwargs else pt.BatchRetrieve(index, wmodel="BM25"))
        secondstage = (kwargs['secondstage'] if 'secondstage' in kwargs else pt.BatchRetrieve(index, wmodel="BM25"))

        rm3_params = {i: kwargs[i] for i in kwargs if i in ['fb_terms', 'fb_docs', 'fb_lambda']}
        weight = (kwargs['weight'] if 'weigth' in kwargs else 0.6)

        # pt.apply.query(lambda row: prevQry(row)) is a rewriter that rewrites the current query with the previous one

        # the next rewriter combines linearly the current query with the previous one linearly
        prev_rewriter = pt.apply.query(lambda row: self.getPrevQuery(row))
        rm3_rewriter = pt.rewrite.RM3(index, **rm3_params)

        # ------------------ rewriter based on RM3 ------------------


        # branch 1: compute the RM3 rewriting of the current query
        branch1 = firststage >> rm3_rewriter
        self.branch1 = branch1
        # branch 2: compute the RM3 rewriting of the previous query
        branch2 = prev_rewriter >> firststage >> rm3_rewriter >> pt.apply.query_0(drop=True)


        rewriter = branch1 >> branch2 >> pt.rewrite.linear(weight, 1 - weight)

        self.rewriter = rewriter

        # join the two rewriting
        self.pipeline = rewriter >> secondstage

    def getConvScores(self, conv, qrels):
        self.prevQueryDict = self.computePrevQueryDict(conv)
        return self.baseExperiment(conv, qrels)
