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
    "id_matrix_simpleTomogravityOD": "matrix_simpleTomogravityOD",

    "id_matrix_avg_generalGravityOD": "matrix_avg_generalGravityOD",
    "id_matrix_std_generalGravityOD": "matrix_std_generalGravityOD",
    "id_matrix_sum_generalGravityOD": "matrix_sum_generalGravityOD",

    "id_matrix_avg_generalTomogravityOD": "matrix_avg_generalTomogravityOD",
    "id_matrix_std_generalTomogravityOD": "matrix_std_generalTomogravityOD",
    "id_matrix_sum_generalTomogravityOD": "matrix_sum_generalTomogravityOD",

    "id_matrix_avg_realOD": "matrix_avg_realOD",
    "id_matrix_std_realOD": "matrix_std_realOD",
    "id_matrix_sum_realOD": "matrix_sum_realOD",

    "id_matrix_avg_simpleGravityOD": "matrix_avg_simpleGravityOD",
    "id_matrix_std_simpleGravityOD": "matrix_std_simpleGravityOD",
    "id_matrix_sum_simpleGravityOD": "matrix_sum_simpleGravityOD",

    "id_matrix_avg_simpleTomogravityOD": "matrix_avg_simpleTomogravityOD",
    "id_matrix_std_simpleTomogravityOD": "matrix_std_simpleTomogravityOD",
    "id_matrix_sum_simpleTomogravityOD": "matrix_sum_simpleTomogravityOD"
}

SOURCE_TYPES = {
    SourceType.BACKBONE_NETWORK
}

AGGREGATIONS = {
    AgreggationType.AGG_5_MINUTES,
    AgreggationType.AGG_10_MINUTES,
    AgreggationType.AGG_1_HOUR,
    AgreggationType.AGG_1_DAY
}
