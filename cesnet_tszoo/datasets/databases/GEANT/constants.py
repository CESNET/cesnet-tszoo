import numpy as np

from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.BACKBONE_NETWORK: "ts_id"
}

DEFAULT_VALUES = {
    'matrix_bandwidth_kbps': 0,
    'average_from_bandwidth_kbps': 0,
    'average_to_bandwidth_kbps': 0,
    'std_from_bandwidth_kbps': 0,
    'std_to_bandwidth_kbps': 0,
    'sum_from_bandwidth_kbps': 0,
    'sum_to_bandwidth_kbps': 0,
    'bandwidth_kbps': 0
}

MATRIX_FEATURE_MAPPINGS = {
    "id_matrix_bandwidth_kbps": "matrix_bandwidth_kbps"
}

SOURCE_TYPES = {
    SourceType.BACKBONE_NETWORK
}

AGGREGATIONS = {
    AgreggationType.AGG_15_MINUTES
}
