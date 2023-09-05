import logging
import dgl

from ge.models.utils.walkers import RandomWalk
from ge.models.deepwalk import DeepWalk


logger = logging.getLogger('ge')


class Node2VecWalk(DeepWalk):
    def __init__(self, g: dgl.DGLGraph, walk_length: int = 200, window: int = 10, emb_size: int = 64,
                 p: float = 1, q: float = 1,
                 batch_size: int = 1e7, epochs: int = 3, n_jobs: int = -1, verbose: bool = True):
        super(Node2VecWalk, self).__init__(
            g=g, walk_length=walk_length, window=window, emb_size=emb_size, batch_size=batch_size,
            epochs=epochs, n_jobs=n_jobs, verbose=verbose
        )
        self.p = p
        self.q = q
        self.iter_path = RandomWalk(
            g=self.g,
            walk_length=self.walk_length,
            p=self.p,
            q=self.q,
            batch_size=self.batch_size
        )
