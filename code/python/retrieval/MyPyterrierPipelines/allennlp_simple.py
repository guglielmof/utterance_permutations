from allennlp_models import pretrained
import logging

from .AbstractPipeline import AbstractPipeline
import pyterrier as pt
from utils import sanitize



class allennlp_simple(AbstractPipeline):

    def __init__(self, index, **kwargs):
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True
        logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
        logging.getLogger('urllib3.connectionpool').disabled = True
        self.name = "allennlp_simple"


        retriever = (kwargs['secondstage'] if 'secondstage' in kwargs else pt.BatchRetrieve(index, wmodel="BM25"))

        self.predictor = pretrained.load_predictor("coref-spanbert")


        # the next rewriter combines linearly the current query with the previous one linearly
        anaphoraResolver = pt.apply.query(lambda row: self.rewrittenUtts[row['qid']])
        self.rewriter = anaphoraResolver
        self.pipeline = self.rewriter >> retriever



    def getConvScores(self, conv, qrels):

        self.rewriteQueries(conv)
        return self.baseExperiment(conv, qrels)


    def rewriteQueries(self, conv):
        self.rewrittenUtts = {}
        for e, r in enumerate(conv.iterrows()):
            _, r = r
            fullConcat = " | ".join(conv['query'].iloc[:e+1])
            doc = self.predictor.coref_resolved(fullConcat)
            self.rewrittenUtts[r['qid']] = sanitize(doc.split(" | ")[-1])


        #print(self.rewrittenUtts)