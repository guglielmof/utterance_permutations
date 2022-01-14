class AbstractContext:


    def computeContext(self, **kwargs):

        raise NotImplementedError


    def computeContextsSimilarity(self, contexts):

        raise NotImplementedError