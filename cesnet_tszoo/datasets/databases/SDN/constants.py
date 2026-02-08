import numpy as np

from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.MININET_SIMULATOR: "ts_id"
}

DEFAULT_VALUES = {
    'matrix_traffic_volume': 0,
    'average_from_traffic_volume': 0,
    'average_to_traffic_volume': 0,
    'std_from_traffic_volume': 0,
    'std_to_traffic_volume': 0,
    'sum_from_traffic_volume': 0,
    'sum_to_traffic_volume': 0,
    'traffic_volume': 0
}

MATRIX_FEATURE_MAPPINGS = {
    "id_matrix_traffic_volume": "matrix_traffic_volume"
}

SOURCE_TYPES = {
    SourceType.MININET_SIMULATOR
}

AGGREGATIONS = {
    AgreggationType.AGG_1_MINUTE
}
