import hashlib
import json
from typing import List, Tuple, ClassVar
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class _Params:
    def hash_id(self):
        return hashlib.md5(json.dumps(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class _DefaultParams:
    src_type_value: ClassVar[str] = 'acct'
    src_id_col: ClassVar[str] = 'src_id'
    tgt_id_col: ClassVar[str] = 'tgt_id'


@dataclass(frozen=True)
class _DataParams(_Params, _DefaultParams):
    graph_table: str = None
    src_col: str = None
    tgt_col: str = None
    src_type_col: str = None
    tgt_type_col: str = None
    entity_id_mapping_table: str = None
    entity_col: str = None
    entity_id_col: str = None
    entity_type_col: str = None
    process_date_col: str = None


@dataclass(frozen=True)
class _FilterParams(_Params):
    start_date: str = None
    end_date: str = None
    assets: List[str] = None
    degrees: List[Tuple[str, int, int]] = None


@dataclass(frozen=True)
class _SaveParams(_Params):
    save_db: str = None
    save_dir_path: str = None


@dataclass(frozen=True)
class Acct2AssetDataParams(_DataParams, _FilterParams, _SaveParams):
    pass


@dataclass(frozen=True)
class Acct2AcctDataParams(_DataParams, _FilterParams, _SaveParams):
    pass

