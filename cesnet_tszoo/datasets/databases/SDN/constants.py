import numpy as np

from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.MININET_SIMULATOR: "ts_id"
}

DEFAULT_VALUES = 0

MATRIX_FEATURE_MAPPINGS = {
    "id_matrix_traffic_volume": "matrix_traffic_volume",

    "id_matrix_avg_traffic_volume": "matrix_avg_traffic_volume",
    "id_matrix_std_traffic_volume": "matrix_std_traffic_volume",
    "id_matrix_sum_traffic_volume": "matrix_sum_traffic_volume",
}

SOURCE_TYPES = {
    SourceType.MININET_SIMULATOR
}

AGGREGATIONS = {
    AgreggationType.AGG_1_MINUTE,
    AgreggationType.AGG_10_MINUTES,
    AgreggationType.AGG_1_HOUR,
    AgreggationType.AGG_1_DAY
}
