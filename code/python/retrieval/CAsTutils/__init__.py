import os
import pandas as pd
import pyterrier as pt
import sys

from pyterrier.io import SUPPORTED_TOPICS_FORMATS
from pyterrier.datasets import DATASET_MAP, RemoteDataset

from functools import partial

from time import time
import numpy as np

os.environ['JAVA_HOME'] = '/ssd/data/faggioli/jdk-11.0.11'

def _read_topics_json(filename, tag='raw_utterance', tokenise=True):
    from jnius import autoclass
    import json
    tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

    data = json.load(open(filename))

    topics = []
    for turn in data:
        turn_id = str(turn['number'])
        for utt in turn['turn']:
            utt_id = str(utt['number'])
            utt_text = utt[tag]
            if tokenise:
                utt_text = " ".join(tokeniser.getTokens(utt_text))
            #            topics.append((turn_id + '_' + utt_id, utt_text, turn_id, utt_id))
            topics.append((turn_id + '_' + utt_id, utt_text))
    #    return pd.DataFrame(topics, columns=["qid", "query", "tid", "uid"])
    return pd.DataFrame(topics, columns=["qid", "query"])




SUPPORTED_TOPICS_FORMATS['json_raw'] = partial(_read_topics_json, tag='raw_utterance')
SUPPORTED_TOPICS_FORMATS['json_manual'] = partial(_read_topics_json, tag='manual_rewritten_utterance')

TREC_CAST = {
    "topics": {
        "original-2019": ("evaluation_topics_v1.0.json",
                          "https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json",
                          "json_raw"),
        "resolved-2019": ("evaluation_topics_annotated_resolved_v1.0.tsv",
                          "https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv",
                          "singleline"),

        "original-2020": ("2020_manual_evaluation_topics_v1.0.json",
                          "https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json",
                          "json_raw"),
        "resolved-2020": ("2020_manual_evaluation_topics_v1.0.json",
                          "https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json",
                          "json_manual"),

    },
    "qrels": {
        "2019": ("cast_eval_topics_2019.qrels", ""),
        "2020": ("cast_eval_topics_2020.qrels", ""),
    },
    "info_url": "https://github.com/daltonj/treccastweb",
}


def loadQueries(query_type='original'):
    print("uploading topics")
    tstart = time()
    sys.stdout.flush()
    DATASET_MAP['cast'] = RemoteDataset("CAST", TREC_CAST)

    all_queries = pt.get_dataset("cast").get_topics(f'{query_type}-2019')
    qrel_queries = pt.get_dataset("cast").get_qrels('2019')['qid'].unique()

    all_queries = all_queries[all_queries['qid'].isin(qrel_queries)]

    all_queries[['topic', 'utterance']] = all_queries['qid'].str.split("_", 1, expand=True)
    print(f"done in {time()-tstart:.3f}s")
    sys.stdout.flush()
    return all_queries

def loadQrels():
    return pt.get_dataset("cast").get_qrels('2019')


def loadIndex(INDEX_PATH='../../../pyterrier/data/index/'):
    print("uploading index")
    tstart = time()
    sys.stdout.flush()
    index = pt.IndexFactory.of(INDEX_PATH)
    print(f"done in {time()-tstart:.3f}s")
    sys.stdout.flush()
    return index