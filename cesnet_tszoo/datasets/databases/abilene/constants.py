from cesnet_tszoo.utils.download import google_drive_url, zenodo_file_url
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

# Secondary sources tried when the DataZoo bucket is unreachable; the files are copies of the bucket content.
FALLBACK_DOWNLOAD_URLS = {
    "Abilene-matrix-backbone_network-10_minutes": [google_drive_url("12fHXNf2uYnSREON7Yw8_9BKF4eZviL9p")],
    "Abilene-matrix-backbone_network-1_day": [google_drive_url("1bQCcVzZZHBnzYmqNFHnO1lx8pfQaN0EN")],
    "Abilene-matrix-backbone_network-1_hour": [google_drive_url("1Q7QnUpfFkWrUTW70kP3ePpXNIJuwraNQ")],
    "Abilene-matrix-backbone_network-5_minutes": [google_drive_url("1MIFskGpn4JBE8bJjxyrnA65SpVZa9I-h")],
    "Abilene-node-backbone_network-10_minutes": [google_drive_url("1YCW4vEO4nSvF3XaaTbbtqcl_hawOY5HJ")],
    "Abilene-node-backbone_network-1_day": [google_drive_url("1WEVPtRM4jcuo7a2h7cQBlskMzMdakh5X")],
    "Abilene-node-backbone_network-1_hour": [google_drive_url("1oPEMuKFmWj6KMSQEaXtmyOjsDQa093Ws")],
    "Abilene-node-backbone_network-5_minutes": [google_drive_url("1HKb6o8RPnJqMCG1Q1iAJ94-FOdvP17j3")],
    "Abilene-node2node-backbone_network-10_minutes": [google_drive_url("1DY9__Ubl-5rsAAc1y8Hf0Wu7iYXjV2OT")],
    "Abilene-node2node-backbone_network-1_day": [google_drive_url("1pap1wlNsolZiU1Pogsc0JgRA11i2wIPM")],
    "Abilene-node2node-backbone_network-1_hour": [google_drive_url("1lobqpUHXaGLTcV6-5PK3GPLIzlHz9mr5")],
    "Abilene-node2node-backbone_network-5_minutes": [google_drive_url("1hYzQ0kh0SWJ2MqxoySeKW09xmeXgoNcI")],
}

# SHA-256 digests of the dataset files (verified after download).
CHECKSUMS = {
    "Abilene-matrix-backbone_network-10_minutes": "4d4d1c46ade9e3aa7857b80735e361a123fedbc1f87fdbd45bf0e696ffe3e8b8",
    "Abilene-matrix-backbone_network-1_day": "39c9cb2188b01a0cfec3d17bd359aac0966d917dc01b032c59dc882f04a20dcf",
    "Abilene-matrix-backbone_network-1_hour": "225b9619e6b3fbae4da33696feec2c12fedf18b17d5f85ffb1f38b1b226afaff",
    "Abilene-matrix-backbone_network-5_minutes": "4fe59c717871621dfbcd4551d731c82eb450f3289473feda82f6dafeb7f3e732",
    "Abilene-node-backbone_network-10_minutes": "1618b57de9baae10347710ec93104b5d847f1f158aae2eb84805a34db3687d0c",
    "Abilene-node-backbone_network-1_day": "114d9476732e2230cd2424aca497ed32cee1272098e1786a5ec933e55e60328b",
    "Abilene-node-backbone_network-1_hour": "016557c1f971e434ef9613ce2854b2968c6b98ff1d0a39932078db2dffac2a8a",
    "Abilene-node-backbone_network-5_minutes": "79249d00a6a783dfd07c9694339cf52933ca578000ec4a053d6b39db648ebff1",
    "Abilene-node2node-backbone_network-10_minutes": "89b02a3d6af1060e40d0ca44cb09e398b1a2c2d2c3316f6d5e54456404b1d922",
    "Abilene-node2node-backbone_network-1_day": "1bdbca2857b3aaa86e32454a8e3dc06cf267d7e00262116f68f5269d2b63a37e",
    "Abilene-node2node-backbone_network-1_hour": "cd93364f8337b2313c5602b0dc22bd526d4771fdf83f508cf840f578fbfc7c57",
    "Abilene-node2node-backbone_network-5_minutes": "50eddcdf696835f7fb088672cc4a0cc63ff76c6aeece12f203fcead9b55e6db9",
}

# Zenodo record https://zenodo.org/records/22721975 (DOI 10.5281/zenodo.22721975, published 2026-10-13) holds the same
# files (byte-identical to the DataZoo bucket, see CHECKSUMS); tried after the Google Drive mirror.
ZENODO_RECORD = 22721975
ZENODO_SOURCES = {name: [zenodo_file_url(ZENODO_RECORD, f"{name}.h5")] for name in CHECKSUMS}
FALLBACK_DOWNLOAD_URLS = {name: list(FALLBACK_DOWNLOAD_URLS.get(name, [])) + ZENODO_SOURCES[name] for name in CHECKSUMS}
