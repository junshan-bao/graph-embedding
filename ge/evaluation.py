import dgl
import numpy as np
import tqdm
import torch
from sklearn.metrics import pairwise_distances


def kg_metrics(g: dgl.DGLGraph, emb: np.array, n_sample: int, batch: int = 100, verbose: bool = True) -> dict:
    rank = list()
    nodes = torch.randperm(g.num_nodes())[:n_sample]

    for idx in tqdm.tqdm(range(n_sample // batch + int(n_sample % batch != 0)), disable=not verbose):
        start, end = idx * batch, (idx + 1) * batch

        dist = pairwise_distances(emb[nodes[start: end]], emb, metric='cosine', n_jobs=-1)
        for node, ds in zip(nodes[start: end], dist):
            ns = g.successors(node)
            nds = ds[ns]
            nds = np.array([nds]) if not isinstance(nds, np.ndarray) else nds
            nds = np.sort(nds)

            for i, d in enumerate(nds):
                r = (ds < d).sum() - i
                rank.append(r)
    ret = {
        'MRR': sum([1 / r for r in rank]) / len(rank),
        'MR': sum(rank) / len(rank),
        'HITS@1': sum([1 if r <= 1 else 0 for r in rank]) / len(rank),
        'HITS@3': sum([1 if r <= 3 else 0 for r in rank]) / len(rank),
        'HITS@10': sum([1 if r <= 10 else 0 for r in rank]) / len(rank),
    }
    return ret


def evaluation(g: dgl.DGLGraph, emb: np.array):
    pass
