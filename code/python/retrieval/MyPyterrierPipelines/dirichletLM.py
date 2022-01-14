from .AbstractPipeline import AbstractPipeline
import pyterrier as pt


class dirichletLM(AbstractPipeline):

    def __init__(self, index):
        self.name = "DirichletLM"
        self.pipeline = pt.BatchRetrieve(index, wmodel="DirichletLM")

