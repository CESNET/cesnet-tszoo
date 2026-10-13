import numpy as np

from cesnet_tszoo.utils.download import google_drive_url, zenodo_file_url
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

# Secondary sources tried when the DataZoo bucket is unreachable; the files are copies of the bucket content.
FALLBACK_DOWNLOAD_URLS = {
    "SDN-matrix-mininet_simulator-10_minutes": [google_drive_url("1mXV_AR3Ty8F8Dbyd83ks47wJxxHI8krs")],
    "SDN-matrix-mininet_simulator-1_day": [google_drive_url("1Bm3FasyLB3E8cJ4cz6VaL6k0oF-NWCcC")],
    "SDN-matrix-mininet_simulator-1_hour": [google_drive_url("11t9f2GB_VyvNsIDMB6t142KoUoDNHEPa")],
    "SDN-matrix-mininet_simulator-1_minute": [google_drive_url("1wxuGLo9fRHQqstpVatJky3kk-uEyTBJd")],
    "SDN-node-mininet_simulator-10_minutes": [google_drive_url("1CrRkVEzULHxG2UcG43GcWFEK7jyJr7AY")],
    "SDN-node-mininet_simulator-1_day": [google_drive_url("1zLqpxWUNjPBFVzBaLbLL20RZ3z-uSrpU")],
    "SDN-node-mininet_simulator-1_hour": [google_drive_url("1ZoPtJkZlozOK9v35MQYijOgNokaE0bKO")],
    "SDN-node-mininet_simulator-1_minute": [google_drive_url("1qjEEfCsPiFFg8g-wA6u_8tJcoH7bQSgE")],
    "SDN-node2node-mininet_simulator-10_minutes": [google_drive_url("1r2UcdxJ3aKxieCHZoa4JcBaBM9JPW0Wx")],
    "SDN-node2node-mininet_simulator-1_day": [google_drive_url("1lyk9g7fY4kUMqznPXrFfy9_MHSYwn9eS")],
    "SDN-node2node-mininet_simulator-1_hour": [google_drive_url("17132gjnyqEQZCs_HQeFJfW4GIhIfkn4j")],
    "SDN-node2node-mininet_simulator-1_minute": [google_drive_url("1MiEaDIcQv_GEJM4hUgobhuLM-XoodraK")],
}

# SHA-256 digests of the dataset files (verified after download).
CHECKSUMS = {
    "SDN-matrix-mininet_simulator-10_minutes": "33e8533c35ea11979d48b6d00e6dd9212578fcc9cdb7cf4f362122499b228f91",
    "SDN-matrix-mininet_simulator-1_day": "e65d03968798a3f1a9232fd62d127a905c5a87fef057b613637b65e19e1e6d45",
    "SDN-matrix-mininet_simulator-1_hour": "0641f111106bceea2ffd72647c65419391023c08a075e01f546cd040d18ff8cb",
    "SDN-matrix-mininet_simulator-1_minute": "3f7d268668188e0bc97e136426c09c230bda6662c2c013cb368e8bd541ea4af1",
    "SDN-node-mininet_simulator-10_minutes": "03ecaa57d46646cd43bf567bd4f913092ffeaf65d2c8dfe3ffe7c2e8af1ec0d0",
    "SDN-node-mininet_simulator-1_day": "d32f3e69a2c1860c3999204a7ee708f764a59c031933b852a296529daf238aaa",
    "SDN-node-mininet_simulator-1_hour": "2149ff280f01bd42a0ac62a2cb147f01d6d0b7f3c3353bdc90d64a08baa8371a",
    "SDN-node-mininet_simulator-1_minute": "1fb7dd5e74f98ef1368958d068c3be8bae125b11b9483c8f97963b0ee960fb23",
    "SDN-node2node-mininet_simulator-10_minutes": "d559053bdefdeef43a555958082fc14b818be22ba79c90b0e731a610d8e4f3b4",
    "SDN-node2node-mininet_simulator-1_day": "ae9d97548dc42047297e5cd6367fc581596a9f1f41b42bed2564b35bac99bd70",
    "SDN-node2node-mininet_simulator-1_hour": "13fec626b92c812793a8c1422fbd431f2a8ae344a23daeacc306147f4f04ce70",
    "SDN-node2node-mininet_simulator-1_minute": "f5ad451273d4fa3f921ada6fcbf66dd1afe4c1283904789666bc7df8e2a34e2d",
}

# Zenodo record https://zenodo.org/records/22721975 (DOI 10.5281/zenodo.22721975, published 2026-10-13) holds the same
# files (byte-identical to the DataZoo bucket, see CHECKSUMS); tried after the Google Drive mirror.
ZENODO_RECORD = 22721975
ZENODO_SOURCES = {name: [zenodo_file_url(ZENODO_RECORD, f"{name}.h5")] for name in CHECKSUMS}
FALLBACK_DOWNLOAD_URLS = {name: list(FALLBACK_DOWNLOAD_URLS.get(name, [])) + ZENODO_SOURCES[name] for name in CHECKSUMS}
