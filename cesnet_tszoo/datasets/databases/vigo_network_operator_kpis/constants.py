import numpy as np

from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.NETWORK_OPERATOR: "ts_id"
}

DEFAULT_VALUES = {
    'scaled_bts': 0,
    'scaled_active_client_sessions': 0
}

SOURCE_TYPES = {
    SourceType.NETWORK_OPERATOR
}

AGGREGATIONS = {
    AgreggationType.AGG_5_MINUTES
}

ADDITIONAL_DATA = {
    "anomalies": (("ts_id", np.uint32), ("start", np.int64), ("end", np.int64)),
    "original_indices": (("ts_id", np.uint32), ("s_ts_id", str), ("r_ts_id", str))
}
