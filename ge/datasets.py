import abc
import logging
from typing import List, Union

import pandas as pd
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

    def get_node_type_mapping(self):
        type_mapping = {a: i + 1 for i, a in enumerate(self.data_params.assets)}
        type_mapping[0] = self.data_params.src_type_value
        return type_mapping

    def get_dgl_graph(self):
        if not self._preprocessed:
            logger.warning('The dataset has not been preprocessed.')
            self.preprocess()
        if self.verbose:
            logger.info('Start to build dgl graph.')

        p = self.data_params

        df = self.db_connector.download_table(p.save_db)

        src_node_data = df[[p.src_id_col]].drop_duplicates().reset_index(drop=True)
        src_node_data[p.tgt_type_col] = p.src_type_value
        src_node_data = src_node_data.rename(columns={p.src_id_col: 'int_node_id', p.tgt_type_col: 'node_type'})

        tgt_node_data = df[[p.tgt_id_col, p.tgt_type_col]].drop_duplicates().reset_index(drop=True)
        tgt_node_data = tgt_node_data.rename(columns={p.tgt_id_col: 'int_node_id', p.tgt_type_col: 'node_type'})

        node_data = pd.concat([src_node_data, tgt_node_data]).reset_index(drop=True)
        nid_mapping = node_data.reset_index().set_index('int_node_id')['index']
        type_mapping = self.get_node_type_mapping()
        node_data['node_type_id'] = node_data['node_type'].map(type_mapping)

        df['dgl_' + p.src_id_col] = df[p.src_id_col].map(nid_mapping)
        df['dgl_' + p.tgt_id_col] = df[p.tgt_id_col].map(nid_mapping)

        u = torch.tensor(df['dgl_' + p.src_id_col].values, dtype=torch.int64)
        v = torch.tensor(df['dgl_' + p.tgt_id_col].values, dtype=torch.int64)

        g = dgl.graph((u, v))
        g.ndata['nid'] = torch.tensor(node_data['int_node_id'].values, dtype=torch.int64)
        g.ndata['tid'] = torch.tensor(node_data['node_type_id'].values, dtype=torch.int64)
        g = dgl.to_bidirected(g, copy_ndata=True)

        if self.verbose:
            logger.info(f'Done: build dgl graph.\nGraph statistics:\n{g}')
        return g


