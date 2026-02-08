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
