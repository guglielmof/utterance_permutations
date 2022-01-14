from . import AbstractExperiment
import experimentalCollections as tc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import json
import pandas as pd
import convLib
from convLib import Annotator
import utils
import time
from retrieval.Analyzers import StandardAnalyzer
import random
from convLib import contexts, Annotator

import experimentalCollections as EC

from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score

from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from collections import Counter

import seaborn as sns
import random

n_max_perms = 10

class ValidPermutations(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_experiment(self):


        permType = "class" #"graph"


        if permType == "class":

            df = pd.read_csv("../../data/measures/permutated_conversations_resolved_100.csv")
            CAsT2019q = df['qid'].unique()
            del df

            uttClasses = pd.read_csv("../../data/test_df_2datasets_new.tsv", delimiter="\t", names=["qid", "query", "class"])
            uttClasses[["topic", "utterance"]] = uttClasses["qid"].str.split("_", 1, expand=True)
            uttClasses["utterance"] = pd.to_numeric(uttClasses["utterance"])
            uttClasses = uttClasses[uttClasses["qid"].isin(CAsT2019q)]

            perms = uttClasses.groupby(["topic"]).apply(getPermutations).values



            nperms = pd.DataFrame([[p[0], len(p[1])] for p in perms], columns=['conversation', 'n permutations'])
            plt.figure(figsize=(18, 12))
            sns.set_context("paper", rc={"font.size": 24, "axes.titlesize": 24, "xtick.labelsize": 24, "ytick.labelsize": 24, "axes.labelsize": 24})
            pal = sns.color_palette("Greens_d", len(nperms))
            rank = nperms["n permutations"].argsort().argsort()
            g = sns.barplot(x='conversation', y='n permutations', data=nperms, palette=np.array(pal[::])[rank])
            for index, row in nperms.iterrows():
                g.text(row.name, row['n permutations']+100, row['n permutations'], color='black', ha="center", fontsize=24)

            plt.savefig("../../data/n_valid_permutations.pdf")
            '''
            with open("../../data/class_perms.txt", "w") as F:
                for pair in perms:
                    F.write(f"{pair[0]}\t{pair[1]}\n")
            '''
def getPreviousSE(utt, uttSet):

    if utt['class'] == "SE":
        return utt["utterance"]

    if utt['class'] == "FT":
        return 1

    else:
        return max([int(i) for i in uttSet[(uttSet["class"].isin(["SE", "FT"])) & (uttSet["utterance"]<utt["utterance"])]["utterance"].values])

def getPermutations(uttSet):
    uttSet["prev"] = uttSet.apply(lambda x: getPreviousSE(x, uttSet), axis=1)

    children = {}

    roots = []

    for idx, row in uttSet[["utterance", "prev"]].iterrows():
        if row["prev"]==row["utterance"]:
            roots.append(row["utterance"])
        else:
            if row["prev"] not in children:
                children[row["prev"]]=[]
            children[row["prev"]].append(row["utterance"])

    print(roots, children)
    permutations = list(set([tuple(uttSet["utterance"].values)] + [permute(roots, children) for _ in range(n_max_perms)]))

    return(uttSet['topic'].unique()[0], permutations)

def permute(roots, children):
    p = []
    currset = roots.copy()
    stack = []
    while len(currset) > 0:
        sidx = random.randint(0, len(currset) - 1)
        selected = currset[sidx]
        print(selected, currset)
        p.append(selected)
        currset = currset[:sidx] + currset[sidx + 1:]
        if selected == 1 and selected in children:
            if len(stack) == 0:
                currset += children[selected]
            else:
                stack[-1] += children[selected]
        elif selected in children:
            stack = [currset] + stack
            currset = children[selected]
        if len(currset) == 0 and len(stack) > 0:
            currset = stack.pop()
    print("_--------------------------")
    return tuple(p)