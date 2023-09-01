import dgl
import torch


class RandomWalk:
    def __init__(self, g: dgl.DGLGraph, walk_length: int, batch_size: int = 1e7):
        self.g = g
        self.walk_length = walk_length
        self.batch_size = int(batch_size)
        self.n_nodes = self.g.num_nodes()

    def _walk(self, nodes=None):
        if nodes is None:
            nodes = torch.randint(0, high=self.n_nodes, size=(self.batch_size,))
        walk_path, _ = dgl.sampling.random_walk(
            g=self.g,
            nodes=nodes,
            length=self.walk_length
        )
        return walk_path

    def __len__(self):
        return self.n_nodes

    def __iter__(self):
        _batch = self.batch_size
        nodes = torch.randperm(self.n_nodes)
        for i in range(self.n_nodes // _batch + int(self.n_nodes % _batch != 0)):
            l, r = i * _batch, (i + 1) * _batch
            for path in self._walk(nodes[l: r]).tolist():
                yield path
