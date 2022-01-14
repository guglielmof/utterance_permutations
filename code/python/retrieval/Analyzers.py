import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import flatten, padded_everygram_pipeline

from collections import Counter
from nltk.stem.porter import *


class StandardAnalyzer():

    def __init__(self, stemmer="porter", stoplist="default_english"):
        nltk.download("stopwords")
        nltk.download("punkt")
        if stoplist=="default_english":
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()

        if stemmer == "porter":
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = lambda x: x

    def analyze(self, document):
        tokens = [self.stemmer.stem(w)
                    for w in word_tokenize(document.lower())
                  if w not in self.stop_words and w.isalnum()]
        return tokens
