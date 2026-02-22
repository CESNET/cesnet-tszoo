from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.BACKBONE_NETWORK: "ts_id"
}

DEFAULT_VALUES = 0

MATRIX_FEATURE_MAPPINGS = {
    "id_matrix_generalGravityOD": "matrix_generalGravityOD",
    "id_matrix_generalTomogravityOD": "matrix_generalTomogravityOD",
    "id_matrix_realOD": "matrix_realOD",
    "id_matrix_simpleGravityOD": "matrix_simpleGravityOD",
    "id_matrix_simpleTomogravityOD": "matrix_simpleTomogravityOD"
}

SOURCE_TYPES = {
    SourceType.BACKBONE_NETWORK
}

AGGREGATIONS = {
    AgreggationType.AGG_5_MINUTES
}
