import abc
import logging
from typing import List, Union

import torch
import dgl

from ge.params import _DataParams, Acct2AssetDataParams
from ge.db_utils import _DataBaseConnector, _CloudStorageConnector
from ge.operators import _Operator, OperatorPipeline


logger = logging.getLogger('ge')


class _Datasets(abc.ABC):
    def __init__(self, data_params: _DataParams, operators: List[Union[_Operator, str, type]],
                 db_connector: _DataBaseConnector = None, cs_connector: _CloudStorageConnector = None,
                 verbose: bool = True):
        self.data_params = data_params
        self.op_pipeline = OperatorPipeline(operators=operators, params=data_params)
        self.db_connector = db_connector
        self.cs_connector = cs_connector
        self.verbose = verbose


class Acct2AssetDatasets(_Datasets):
    def __init__(self, data_params: Acct2AssetDataParams, operators: List[Union[_Operator, str, type]],
                 db_connector: _DataBaseConnector = None, cs_connector: _CloudStorageConnector = None,
                 verbose: bool = True):
        super().__init__(data_params=data_params, operators=operators,
                         db_connector=db_connector, cs_connector=cs_connector,
                         verbose=verbose)
        self._preprocessed = False

    @property
    def preprocess_query(self):
        return self.op_pipeline.call()

    def preprocess(self):
        if self.verbose:
            logger.info('Start to fetch and preprocess graph.')
        self.db_connector.create_table(
            table_name=self.data_params.save_db,
            table_query=self.preprocess_query,
            verbose=self.verbose,
        )
        self._preprocessed = True
        if self.verbose:
            logger.info('Done: fetch and preprocess graph.')
        return self

    def to_graph(self):
        if not self._preprocessed:
            logger.warning('The dataset has not been preprocessed.')
            self.preprocess()
        if self.verbose:
            logger.info('Start to build dgl graph.')

        p = self.data_params

        df = self.db_connector.download_table(p.save_db)
        df[p.src_col]
        u = torch.tensor(df[self.data_params.src_col].values, dtype=torch.int64)
        v = torch.tensor(df[self.data_params.tgt_col].values, dtype=torch.int64)

        g = dgl.graph((u, v))
        g = dgl.to_bidirected(g)

        if self.verbose:
            logger.info(f'Done: build dgl graph.\nGraph statistics:\n{g}')
        return g


