from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.SERVICE_PROVIDER: "ts_id"
}

DEFAULT_VALUES = 0

SOURCE_TYPES = {
    SourceType.SERVICE_PROVIDER
}

AGGREGATIONS = {
    AgreggationType.AGG_10_MINUTES,
    AgreggationType.AGG_1_HOUR,
    AgreggationType.AGG_1_DAY
}
