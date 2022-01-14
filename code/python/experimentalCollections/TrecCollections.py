import pytrec_eval
import time
import os
from utils import chunk_based_on_number
from multiprocessing import Pool
import json


class AbstractCollection:

    def __init__(self, logger=None):
        self.logger = logger

    def importCollection(self, nThreads=1, selected_runs=None):
        if not self.logger is None:
            self.logger.info("importing the collection")
            stime = time.time()
        self.runs = self.import_runs(nThreads, selected_runs=selected_runs)
        self.qrels = self.import_qrels()

        self.systems = list(self.runs.keys())
        self.topics = list(self.qrels.keys())

        if not self.logger is None:
            self.logger.info(f"collection imported in {time.time() - stime:.2f} seconds")

        return self

    def import_qrels(self, qPath=None):
        # -------------------------- IMPORT QRELS -------------------------- #
        if qPath is None:
            qPath = self.qrel_path

        with open(qPath, "r") as F:
            qrels = pytrec_eval.parse_qrel(F)

        return qrels

    def import_runs(self, nThreads, selected_runs=None):

        systems_paths = os.listdir(self.runs_path)
        if selected_runs is not None:
            systems_paths = [s for s in systems_paths if ".".join(s.split(".")[:-1]) in selected_runs]
        # -------------------------- IMPORT RUNS -------------------------- #
        runs = {}
        chunks = chunk_based_on_number(systems_paths, nThreads)

        with Pool(processes=nThreads) as pool:

            futuresRunsDict = [pool.apply_async(getPartialRuns, [chunk, self.runs_path]) for chunk in chunks]
            runsDict = [res.get() for res in futuresRunsDict]

        for d in runsDict:
            for r in d:
                runs[r] = d[r]

        return runs



class RQV04(AbstractCollection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_path = "../../../21-ECIR-CFFSZ/data/experiment/"
        self.runs_path = self.data_path + "runs/RQV04/"
        self.qrel_path = self.data_path + "pool/RQV04/expanded_robust04.qrels"

    def getTitleQueries(self):
        queries = list(self.import_qrels().keys())
        titles = []
        for q in queries:
            _, g, u = q.split("-")
            if g == "50" and u == "1":
                titles.append(q)

        return titles


# used to parallelize the import of the runs
def getPartialRuns(rNamesList, runs_path):
    runs = {}
    for e, run_filename in enumerate(rNamesList):
        with open(runs_path + run_filename, "r") as F:
            try:
                runs[".".join(run_filename.split(".")[:-1])] = pytrec_eval.parse_run(F)
            except Exception as e:
                print(e)
                print(run_filename)
    return runs



class conv_collection(AbstractCollection):

    def importCollection(self, conv=False, nconv=-1, nThreads=1, selected_runs=None):

        if not self.logger is None:
            self.logger.info("importing the collection")
            stime = time.time()

        self.runs = self.import_runs(nThreads, selected_runs=selected_runs)
        self.qrels_tr = self.import_qrels(self.qrel_tr_path)
        self.qrels_ts = self.import_qrels(self.qrel_ts_path)

        self.systems = list(self.runs.keys())

        self.topics_tr = list(self.qrels_tr.keys())
        self.topics_ts = list(self.qrels_ts.keys())

        self.conv2utt_tr = {}
        self.conv2utt_ts = {}

        for t in self.topics_tr:
            tid, uid = t.split("_")
            if tid not in self.conv2utt_tr:
                self.conv2utt_tr[tid] = []
            self.conv2utt_tr[tid].append(t)

        for t in self.topics_ts:
            tid, uid = t.split("_")
            if tid not in self.conv2utt_ts:
                self.conv2utt_ts[tid] = []
            self.conv2utt_ts[tid].append(t)

        if conv:
            self.conv_tr = self.import_conv(self.conv_tr_path)
            self.conv_ts = self.import_conv(self.conv_ts_path)

            self.conv_ts_resolved = {}
            with open(self.conv_ts_resolved_path, "r") as F:
                for l in F.readlines():
                    idu, text = l.strip().split("\t")
                    self.conv_ts_resolved[idu] = text

        if not self.logger is None:
            self.logger.info(f"collection imported in {time.time() - stime:.2f} seconds")

        return self

    def import_conv(self, path):
        with open(path) as file:
            return json.load(file)

    def evalRuns(self, mLabel):
        if not self.logger is None:
            self.logger.info("computing measures...")
            stime = time.time()
        topic_evaluator = pytrec_eval.RelevanceEvaluator(self.qrels_ts, {mLabel})
        self.measure = {s: topic_evaluator.evaluate(self.runs[s]) for s in self.systems}

        # ---- remove the measure keyword from the measure dictionary
        self.measure = {r: {t: self.measure[r][t][mLabel] for t in self.measure[r]} for r in self.measure}

        if not self.logger is None:
            self.logger.info(f"done in {time.time() - stime:.2f} seconds")


        return self


class CAsT(conv_collection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        '''
		The collection has some problems:
		1) MARCO_5089548
		   MARCO_4867704
		   are duplicates for topic 4_4

		2) input.UDInfoC_BL is illformed (duplicate row)

		3) input.UDInfoC_TS has scores between apexes

		'''

        self.data_path = "../../../data/TREC/TREC_28_2019_CAsT/"

        self.runs_path = self.data_path + "runs/"

        self.qrel_tr_path = self.data_path + "qrels/training/train_topics_mod.qrel.txt"
        self.qrel_ts_path = self.data_path + "qrels/test/2019qrels.txt"

        self.conv_tr_path = self.data_path + "topics/training/train_topics_v1.0.json"
        self.conv_ts_path = self.data_path + "topics/test/evaluation_topics_v1.0.json"

        self.conv_ts_resolved_path = self.data_path + "topics/test/evaluation_topics_annotated_resolved_v1.0.tsv"

        self.manual_runs = ["input.combination", "input.datasetreorder", "input.rerankingorder", "input.topicturnsort",
                            "input.humanbert", "input.CFDA_CLIP_RUN6", "input.CFDA_CLIP_RUN1", "input.CFDA_CLIP_RUN8",
                            "input.manual_indri", "input.VESBERT", "input.VESBERT1000", "input.ug_cur_sdm",
                            "input.UMASS_DMN_V2", "input.ict_wrfml", "input.UNH-trema-ecn", "input.unh-trema-relco",
                            "input.UNH-trema-ent", "input.clacMagic", "input.clacMagicRerank", "input.RUCIR-run1",
                            "input.h2oloo_RUN3", "input.h2oloo_RUN4", "input.h2oloo_RUN5",]
