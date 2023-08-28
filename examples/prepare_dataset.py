from ge.datasets import Acct2AssetDataParams
from ge.datasets import Acct2AssetDatasets
from ge.operators import (
    NullFilter,
    DateFilter,
    DropDuplicatesFilter,
    AssetFilter,
    AssetDegreeFilter,
    Acct2AssetMappingIdOperator
)
from ge.db_utils import (
    SnowflakeConnector,
    GCloudStorageConnector
)


if __name__ == '__main__':
    param = Acct2AssetDataParams(
        graph_table='AP_CUR_R_FEATSCI.CURATED_FEATURE_SCIENCE_RED.acct2asset_alm',
        src_col='acct',
        tgt_col='asset_val',
        asset_type_col='asset_name',
        entity_id_mapping_table='AP_CUR_R_FEATSCI.CURATED_FEATURE_SCIENCE_RED.acct2asset_entity_id_map_alm',
        entity_col='raw_node_id',
        entity_id_col='int_node_id',
        entity_type_col='node_typ',
        process_date_col='par_process_date',
        start_date='2022-05-28',
        end_date='2022-06-01',
        assets=[
            'device_id',
            'mobile',
            'merchant_side_email',
            'card_id',
            'shipping_mobile',
            'billing_mobile',
            'consumer_address_norm',
            'shipping_address_norm',
            'billing_address_norm',
        ],
        degrees=[
            ('device_id', 1, 200),
            ('mobile', 1, 200),
            ('merchant_side_email', 1, 10),
            ('card_id', 1, 200),
            ('shipping_mobile', 1, 10),
            ('billing_mobile', 1, 10),
            ('consumer_address_norm', 1, 10),
            ('shipping_address_norm', 1, 10),
            ('billing_address_norm', 1, 10),
        ],
        save_db='ap_cur_frdrisk_g.public.junshan_dev_graph_embeddding_graph_20220601',
        save_dir_path='gs://afterpay-fraud-ml-ds-cash-production/junshan/graph-embedding/graph_20220601'
    )

    dataset = Acct2AssetDatasets(
        data_params=param,
        filters=[
            NullFilter,
            DateFilter,
            DropDuplicatesFilter,
            AssetFilter,
            AssetDegreeFilter,
            Acct2AssetMappingIdOperator
        ],
        db_connector=SnowflakeConnector(),
        cs_connector=GCloudStorageConnector(),
    )

    dataset.process()



