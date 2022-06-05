from metrics import e_recall, c_recall
import numpy as np
import faiss, torch, copy
from sklearn.preprocessing import normalize
from tqdm import tqdm

def select(metricname):
    #### Metrics based on euclidean distances
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)

    #### Metrics based on cosine similarity
    elif 'c_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return c_recall.Metric(k)
    else:
        raise NotImplementedError("Metric {} not available!".format(metricname))
