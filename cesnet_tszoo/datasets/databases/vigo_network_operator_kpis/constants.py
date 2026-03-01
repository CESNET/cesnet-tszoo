import numpy as np

from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.NETWORK_OPERATOR: "ts_id"
}

DEFAULT_VALUES = 0

SOURCE_TYPES = {
    SourceType.NETWORK_OPERATOR
}

AGGREGATIONS = {
    AgreggationType.AGG_5_MINUTES,
    AgreggationType.AGG_10_MINUTES,
    AgreggationType.AGG_1_HOUR,
    AgreggationType.AGG_1_DAY
}
