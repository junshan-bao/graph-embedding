import abc
import logging
from typing import List, Union

from ge.params import _DataParams, Acct2AssetDataParams
from ge.db_utils import _DataBaseConnector, _CloudStorageConnector
from ge.operators import _Operator, OperatorPipeline


logger = logging.getLogger('ge')


class _Datasets(abc.ABC):
    def __init__(self, data_params: _DataParams, operators: Union[List[_Operator], List[type]],
                 db_connector: _DataBaseConnector = None, cs_connector: _CloudStorageConnector = None,
                 verbose: bool = True):
        self.data_params = data_params
        self.op_pipeline = OperatorPipeline(operators=operators, params=data_params)
        self.db_connector = db_connector
        self.cs_connector = cs_connector
        self.verbose = verbose


class Acct2AssetDatasets(_Datasets):
    def __init__(self, data_params: Acct2AssetDataParams, operators: Union[List[type], List[_Operator]],
                 db_connector: _DataBaseConnector = None, cs_connector: _CloudStorageConnector = None,
                 verbose: bool = True):
        super().__init__(data_params=data_params, operators=operators,
                         db_connector=db_connector, cs_connector=cs_connector,
                         verbose=verbose)

    def process(self):
        if self.verbose:
            logger.info('Start to fetch and process graph.')
        graph_query = self.op_pipeline.call()

        self.db_connector.create_table(
            table_name=self.data_params.save_db,
            table_query=graph_query,
            verbose=self.verbose,
        )
