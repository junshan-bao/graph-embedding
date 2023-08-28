import hashlib
import json
from typing import List, Tuple
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class _Params:
    def hash_id(self):
        return hashlib.md5(json.dumps(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class _DataParams(_Params):
    graph_table: str = None
    src_col: str = None
    tgt_col: str = None
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
    acct_type_col: str = None
    asset_type_col: str = None


@dataclass(frozen=True)
class Acct2AcctDataParams(_DataParams, _FilterParams, _SaveParams):
    pass



