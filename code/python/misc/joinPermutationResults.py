import os
import pandas as pd
import sys

df = pd.DataFrame()

basepath = "../../data/measures/"


for fn in os.listdir(basepath):

    if "permutated_conversations" in fn and f"_{sys.argv[1]}_{sys.argv[2]}" in fn:

        _,_,cId,_,_ = fn.split("_")
        tmpDf = pd.read_csv(f"{basepath}{fn}")

        with open(f"{basepath}permutations_{cId}_{sys.argv[1]}_{sys.argv[2]}.csv", "r") as F:
            permList = [l.strip().split(",") for l in F.readlines()]
            getPos = lambda x: permList[x['perm']].index(x['qid'])
            tmpDf['order'] = tmpDf.apply(getPos, axis=1)

        df = df.append(tmpDf)
print(df)
df.to_csv(f"{basepath}permutated_conversations_{sys.argv[1]}_{sys.argv[2]}.csv")