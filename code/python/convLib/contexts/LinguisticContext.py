from .AbstractContext import AbstractContext


from collections import Counter


import numpy as np
from sklearn.preprocessing import normalize


class LinguisticContext(AbstractContext):

    def __init__(self, documents, corpus, analyzer):


        self.computeContext(documents=documents, corpus=corpus, analyzer=analyzer)

    def computeContext(self, **kwargs):
        analyzer = kwargs['analyzer']
        corpus = kwargs['corpus']
        documents = kwargs['documents']
        tokenized_documents = [t for d in documents if d in corpus for t in analyzer.analyze(corpus[d])]
        self.contextRepr = Counter(tokenized_documents)
        return self

    def computeContextsSimilarity(self, contexts):
        # Get the entire vocabulary for all queries
        vocab = set(self.contextRepr.keys()).union(*[set(c.contextRepr.keys()) for c in contexts])
        # associate to each word an unique integer (position)
        w2p = {w: e for e, w in enumerate(vocab)}

        arrayContext = np.zeros((1, len(vocab)))

        for w in self.contextRepr:
            arrayContext[0, w2p[w]] = self.contextRepr[w]

        # convert the language models in a matrix
        matrixContexts = np.zeros((len(contexts), len(vocab)))

        for e, c in enumerate(contexts):
            for w in c.contextRepr:
                matrixContexts[e, w2p[w]] = c.contextRepr[w]


        # normalize the matrix with l2 to easily compute the cosine similarity
        arrayContext = normalize(arrayContext)
        matrixContexts = normalize(matrixContexts)
        simVector = np.matmul(arrayContext, matrixContexts.T)

        return simVector[0]

