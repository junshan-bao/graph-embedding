import dgl
import torch
import tqdm
from typing import Union, List


class RandomWalk:
    def __init__(self, g: dgl.DGLGraph, walk_length: int, batch_size: int = 1e7, path_type: type = int,
                 p: float = 1, q: float = 1, node_type: [List[str], None] = None,
                 verbose: bool = False):
        self.g = g
        self.walk_length = walk_length
        self.batch_size = int(batch_size)
        self.path_type = path_type
        self.p = p
        self.q = q
        self.node_type = node_type
        self.verbose = verbose
        self.n_nodes = self.g.num_nodes()

    def _walk(self, nodes):
        walk_path = dgl.sampling.node2vec_random_walk(
            g=self.g,
            nodes=nodes,
            p=self.p,
            q=self.q,
            walk_length=self.walk_length
        )
        return walk_path

    def to_txt(self, path):
        _type = self.path_type
        self.path_type = str
        with open(path, 'w', encoding='utf-8') as f:
            for line in tqdm.tqdm(self, disable=not self.verbose):
                f.write(' '.join(line) + '\n')

    def __len__(self):
        return self.n_nodes

    def __iter__(self):
        _batch = self.batch_size
        nodes = torch.randperm(self.n_nodes)

        for i in range(self.n_nodes // _batch + int(self.n_nodes % _batch != 0)):
            l, r = i * _batch, (i + 1) * _batch
            for path in self._walk(nodes[l: r]).tolist():
                if self.path_type != int:
                    yield [self.path_type(p) for p in path]
                else:
                    yield path
