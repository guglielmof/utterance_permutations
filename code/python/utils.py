import math

import matplotlib.colors as colors
import numpy as np

def recursiveSplitGop(gopScores, delimiter="_"):
        names = list(gopScores.keys())
        lNames = names[0].split(delimiter)
        if len(lNames) == 1:
            return gopScores
        else:
            newGop = {}
            for n in names:
                name_splitted = n.split("_")
                if name_splitted[0] not in newGop:
                    newGop[name_splitted[0]] = {}
                newGop[name_splitted[0]][delimiter.join(name_splitted[1:])] = gopScores[n]

            newGop = {n:recursiveSplitGop(newGop[n]) for n in newGop}
            return newGop



def getUniqueFactors(gopScores, delimiter="_"):
    names = list(gopScores.keys())
    factors = [{n} for n in names[0].split(delimiter)]
    for n in names:
        for e, factor in enumerate(n.split(delimiter)):
            factors[e].add(factor)

    return factors

def chunk_based_on_number(lst, chunk_numbers):
    n = math.ceil(len(lst) / chunk_numbers)

    chunks = []
    for x in range(0, len(lst), n):
        each_chunk = lst[x: n + x]

        #if len(each_chunk) < n:
        #    each_chunk = each_chunk + [None for y in range(n - len(each_chunk))]
        chunks.append(each_chunk)

    return chunks

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap