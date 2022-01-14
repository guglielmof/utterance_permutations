from .AbstractPipeline import AbstractPipeline
import pyterrier as pt


class bm25(AbstractPipeline):

    def __init__(self, index):
        self.name = "BM25"
        self.pipeline = pt.BatchRetrieve(index, wmodel="BM25")

