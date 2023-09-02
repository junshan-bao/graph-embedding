import time
import logging
import multiprocessing as mp
import pickle
import pathlib
import dgl
import tqdm
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from ge.models.utils.walkers import RandomWalk
from ge.utils import check_and_mkdir

logger = logging.getLogger('ge')


class Callback(CallbackAny2Vec):
    def __init__(self, epoch=0, last_total_loss=0, verbose=True):
        self.epoch = epoch
        self.last_total_loss = last_total_loss
        self.verbose = verbose
        self.start_time = time.time()

    def on_epoch_begin(self, model):
        self.start_time = time.time()
        if self.verbose:
            logger.info('Start epoch {}'.format(self.epoch))

    def on_epoch_end(self, model):
        if self.verbose:
            logger.info(f'Finish epoch {self.epoch}. Time cost {time.time() - self.start_time: .2f}')
        curr_total_loss = model.get_latest_training_loss()
        loss = curr_total_loss if self.last_total_loss == 0 else curr_total_loss - self.last_total_loss
        self.last_total_loss = curr_total_loss
        if self.verbose:
            logger.info('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1


class DeepWalk:
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
        self.model = Word2Vec(
            vector_size=self.emb_size,
            sg=1,
            hs=1,
            workers=self.n_jobs,
            window=self.window,
            epochs=self.epochs,
            min_count=1,
        )

    def _build_vocab(self):
        if self.verbose:
            logger.info('Start to build vocab')
        start_time = time.time()
        self.model.build_vocab(corpus_iterable=tqdm.tqdm(self.iter_path, disable=not self.verbose))
        if self.verbose:
            logger.info(f'Finish to build vocab, time costs {time.time() - start_time:.2f}')

    def train(self):
        self._build_vocab()
        if self.verbose:
            logger.info('Start to train model')
        start_time = time.time()
        self.model.train(
            tqdm.tqdm(self.iter_path, disable=not self.verbose),
            total_examples=len(self.iter_path),
            epochs=self.epochs,
            queue_factor=10,
            callbacks=[Callback(verbose=self.verbose)]
        )
        if self.verbose:
            logger.info(f'Finish to train model, time costs {time.time() - start_time:.2f}')
        return self

    def get_embedding(self):
        emb = torch.zeros((self.g.num_nodes(), self.emb_size))
        for i in range(self.g.num_nodes()):
            emb[i, :] = self.model.wv[i]
        return emb

    def save_embedding(self, path):
        emb = self.get_embedding()

        check_and_mkdir(path)
        with open(path, 'wb') as f:
            pickle.dump(emb, f)
        return self

    def save_model(self, path):
        check_and_mkdir(path)

        self.model.save(path)

    def load_model(self, path):
        self.model = Word2Vec.load(path)
