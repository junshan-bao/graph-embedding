import re
import abc
import logging
from typing import List, Tuple, Union

from ge.params import _DataParams, Acct2AssetDataParams

logger = logging.getLogger('ge')


class _Operator:
    def __init__(self, params: _DataParams):
        self.params = params
        if not self.check_params():
            logger.warning(f'Params do not fit with {self.name}, the program will skip it.')

    @property
    def name(self):
        return '_'.join([n.lower() for n in re.findall('[A-Z][a-z]*', self.__class__.__name__)])

    @abc.abstractmethod
    def check_params(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def call(self, depend_table=None):
        raise NotImplementedError


class NullFilter(_Operator):
    def __init__(self, params: _DataParams):
        super().__init__(params=params)

    def check_params(self) -> bool:
        if not hasattr(self.params, 'src_col') or not hasattr(self.params, 'tgt_col'):
            return False
        return True

    def call(self, depend_table=None):
        query = f"""
            {self.name} as (
                select *
                from {depend_table or self.params.graph_table}
                where {self.params.src_col} is not null and {self.params.src_col} <> ''
                and {self.params.tgt_col} is not null and {self.params.tgt_col} <> ''
            )
        """
        return query


class DateFilter(_Operator):
    def __init__(self, params: _DataParams):
        super().__init__(params=params)

    def check_params(self) -> bool:
        if self.params.process_date_col is None \
                or self.params.start_date is None \
                or self.params.end_date is None:
            return False
        return True

    def call(self, depend_table=None):
        query = f"""
            {self.name} as (
                select *
                from {depend_table or self.params.graph_table}
                where {self.params.process_date_col} >= '{self.params.start_date}'
                and {self.params.process_date_col} < '{self.params.end_date}'
            )
        """
        return query


class AssetFilter(_Operator):
    def __init__(self, params: Acct2AssetDataParams):
        super().__init__(params=params)

    def check_params(self) -> bool:
        if not isinstance(self.params, Acct2AssetDataParams) or self.params.assets is None:
            return False
        return True

    def call(self, depend_table=None):
        assets_query = '({})'.format(
            ', '.join("'{}'".format(asset) for asset in self.params.assets)
        )
        query = f"""
            {self.name} as (
                select *
                from {depend_table or self.params.graph_table}
                where {self.params.tgt_col} in {assets_query}
            )
        """
        return query


class AssetDegreeFilter(_Operator):
    def __init__(self, params: Acct2AssetDataParams):
        super().__init__(params=params)

    def check_params(self) -> bool:
        if not isinstance(self.params, Acct2AssetDataParams) or self.params.degrees is None:
            return False
        if self.params.asset_type_col is None:
            return False
        return True

    def call(self, depend_table=None):
        filter_asset_script_list = []
        for asset, min_degree, max_degree in self.params.degrees:
            _script = f"""
                select {self.params.asset_type_col}
                       ,{self.params.tgt_col}
                from
                (
                    select {self.params.asset_type_col}
                           ,{self.params.tgt_col}
                           ,count({self.params.src_col}) cnt
                    from {depend_table or self.params.graph_table}
                    where {self.params.asset_type_col} = '{asset}'
                    group by {self.params.asset_type_col}, {self.params.tgt_col}
                    having cnt >= {min_degree} and cnt <= {max_degree}
                ) t
            """
            filter_asset_script_list.append(_script)
        filter_asset_script = '\n\t\tunion all\n'.join(filter_asset_script_list)

        query = f"""
            temp_{self.name} as (
                {filter_asset_script}
            ), {self.name} as (
                select t1.*
                from {depend_table or self.params.graph_table} as t1
                inner join temp_{self.name}  as t2 
                on t1.{self.params.asset_type_col}=t2.{self.params.asset_type_col}
                and t1.{self.params.tgt_col}=t2.{self.params.tgt_col}
            )
        """
        return query


class DropDuplicatesFilter(_Operator):
    def __init__(self, params: _DataParams):
        super().__init__(params=params)

    def check_params(self) -> bool:
        return True

    def call(self, depend_table=None):
        group_by_list = [self.params.src_col, self.params.tgt_col]
        if hasattr(self.params, 'asset_type_col') and self.params.asset_type_col is not None:
            group_by_list.append(self.params.asset_type_col)

        query = f"""
            {self.name} as (
                select {', '.join(group_by_list)}
                from {depend_table or self.params.graph_table}
                group by {', '.join(group_by_list)}
            )
        """
        return query


class Acct2AssetMappingIdOperator(_Operator):
    def __init__(self, params: Acct2AssetDataParams):
        super().__init__(params=params)

    def check_params(self) -> bool:
        if self.params.entity_id_mapping_table is None:
            return False
        return True

    def call(self, depend_table=None):
        query = f"""
            {self.name} as (
                select t1.*
                       ,t2.{self.params.entity_id_col} as src_id
                       ,t3.{self.params.entity_id_col} as tgt_id
                from {depend_table or self.params.graph_table} as t1
                inner join {self.params.entity_id_mapping_table} as t2
                on t2.{self.params.entity_type_col}='acct' 
                and t1.{self.params.src_col}=t2.{self.params.entity_col}
                inner join {self.params.entity_id_mapping_table} as t3
                on t1.{self.params.asset_type_col}=t3.{self.params.entity_type_col} 
                and t1.{self.params.tgt_col} = t3.{self.params.entity_col}
            )
        """

        return query


class OperatorPipeline:
    def __init__(self, filters: Union[List[_Operator], List[type]], params: _DataParams = None):
        self.filters = filters
        self.params = params

    def call(self):
        query = list()
        last_filter_name = self.params.graph_table
        for f in self.filters:
            if not isinstance(f, _Operator):
                f = f(self.params)
            if f.check_params():
                query.append(f.call(last_filter_name))
                last_filter_name = f.name

        start_query = 'with' if len(query) > 0 else ''
        op_query = ', '.join(query)
        select_query = f"""
            select *
            from {last_filter_name}
        """

        final_query = start_query + op_query + select_query
        return final_query
