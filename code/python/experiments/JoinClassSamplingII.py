from . import AbstractExperiment

import numpy as np

import pandas as pd



class JoinClassSamplingII(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):
        conversations = ['31', '32', '33', '34', '37', '40', '49', '50', '54', '56', '58', '59', '61', '67', '68', '69',
                         '75', '77', '78', '79']
        #conversations = ['31', '32', '34', '37', '40', '49', '50', '54', '56', '58', '59', '61', '67', '68', '69',
        #                 '75', '77', '79']
        ds = []
        for c in conversations:
            ds.append(pd.read_csv(f"../../data/measures/{c}.csv"))

        measures = pd.concat(ds)

        measures[['topic', 'utterance']] = measures['qid'].str.split("_", 1, expand=True)

        measures['utterance'] = pd.to_numeric(measures['utterance'])

        measures = measures.groupby(['name', 'topic', 'perm', 'qtype'])\
                    .apply(lambda x: get_order(x, measures))\
                    .reset_index()\
                    .drop("level_4", axis=1)


        avgs = measures.groupby(['name', 'topic', 'perm', 'qtype']).aggregate("mean").reset_index()
        topicAggregations = avgs[['name', 'topic', 'nDCG@3', 'qtype']].groupby(['name', 'topic', 'qtype']).aggregate(["min", "mean", "max", "std"]).reset_index()
        #with pd.option_context('display.max_rows', None, 'display.max_columns', df.shape[1]):
        with pd.option_context('display.max_rows', None):
            print(topicAggregations)
        #avgs[['name', 'perm', 'nDCG@3', 'qtype']].groupby(['name', 'perm', 'qtype']).aggregate("mean").reset_index())


        measures.to_csv(f"../../data/measures/full_valid_100.csv")
        '''
        sampled = measures[(measures['topic']=='31') & (measures['perm']==31)]

        sampled.groupby(['name', 'perm']).apply(buildMatrices)
        '''
def get_order(ds, measure):
    ds2 = ds.copy()
    if ds2['name'].unique()[0] =='RM3_seq':
        ds2['order'] = np.arange(len(ds2.index))

    else:
        ds2 = ds2.sort_values('utterance')
        refOrder = measure[(measure['name']=='RM3_seq') &
                           (measure['topic']==ds['topic'].unique()[0]) &
                           (measure['perm']==ds['perm'].unique()[0])&
                           (measure['qtype']==ds['qtype'].unique()[0])]['utterance'].values


        refOrder = [int(r) for r in refOrder]


        new_order = np.argsort(refOrder)
        ds2['order'] = new_order
        ds2 = ds2.sort_values('order')

    return ds2[['qid', 'nDCG@3', 'utterance']]



def buildMatrices(ds):
    print(ds)

    x = np.zeros((len(ds.index), len(ds.index)))
    idxs = list(np.argsort(ds['order'].values))
    print(np.ix_(idxs[:-1],idxs[1:]))
    x[np.ix_(idxs[:-1],idxs[1:])] = 1

    print(x)
    print(idxs)