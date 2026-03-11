import numpy as np

from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.BACKBONE_NETWORK: "ts_id"
}

DEFAULT_VALUES = 0

MATRIX_FEATURE_MAPPINGS = {
    "id_matrix_bandwidth_kbps": "matrix_bandwidth_kbps",

    "id_matrix_avg_bandwidth_kbps": "matrix_avg_bandwidth_kbps",
    "id_matrix_std_bandwidth_kbps": "matrix_std_bandwidth_kbps",
    "id_matrix_sum_bandwidth_kbps": "matrix_sum_bandwidth_kbps"
}

SOURCE_TYPES = {
    SourceType.BACKBONE_NETWORK
}

AGGREGATIONS = {
    AgreggationType.AGG_15_MINUTES,
    AgreggationType.AGG_1_HOUR,
    AgreggationType.AGG_1_DAY
}
