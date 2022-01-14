from .AbstractPipeline import AbstractPipeline
import pyterrier as pt


class rm3(AbstractPipeline):

    def __init__(self, index, **kwargs):
        self.name = "RM3"

        firststage = (kwargs['firststage'] if 'firststage' in kwargs else pt.BatchRetrieve(index, wmodel="BM25"))
        secondstage = (kwargs['secondstage'] if 'secondstage' in kwargs else pt.BatchRetrieve(index, wmodel="BM25"))

        rm3_params = {i:kwargs[i] for i in kwargs if i in ['fb_terms', 'fb_docs', 'fb_lambda']}
        
        self.pipeline = firststage >> pt.rewrite.RM3(index, **rm3_params) >> secondstage

