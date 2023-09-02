import time
import logging
import multiprocessing as mp
import dgl
import tqdm
import torch

from ge.models.utils.walkers import RandomWalk

logger = logging.getLogger('ge')


class Fasttext:
    def __init__(self, g: dgl.DGLGraph, walk_length: int = 200, window: int = 10, emb_size: int = 64,
                 batch_size: int = 1e7, epochs: int = 3, n_jobs: int = -1, verbose: bool = True):
        self.g = g
        self.walk_length = walk_length
        self.emb_size = emb_size
        self.window = window
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count() - 1
        self.verbose = verbose
        self.iter_path = RandomWalk(self.g, walk_length=self.walk_length, batch_size=self.batch_size)
        self.model = None
    def train(self):
        if self.verbose:
            logger.info('Start to train model')
        start_time = time.time()

        if self.verbose:
            logger.info(f'Finish to train model, time costs {time.time() - start_time:.2f}')
        return self

    def get_embedding(self):
        emb = torch.zeros((self.g.num_nodes(), self.emb_size))
        for i in range(100):
            emb[i, :] = self.model.wv[i]
        return emb

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.load(path)
