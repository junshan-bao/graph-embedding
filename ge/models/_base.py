import abc
import pickle


class _Model:
    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, path):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, path):
        raise NotImplementedError

    @abc.abstractmethod
    def save_embedding(self, path):
        raise NotImplementedError

    @staticmethod
    def load_embedding(path):
        with open(path, 'rb') as f:
            emb = pickle.load(f)
        return emb
