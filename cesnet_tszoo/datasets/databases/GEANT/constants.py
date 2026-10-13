import numpy as np

from cesnet_tszoo.utils.download import google_drive_url, zenodo_file_url
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

# Secondary sources tried when the DataZoo bucket is unreachable; the files are copies of the bucket content
# as published on 2026-09-10. Replace both the ids and the digests when the bucket is re-uploaded.
FALLBACK_DOWNLOAD_URLS = {
    "GEANT-matrix-backbone_network-15_minutes": [google_drive_url("1xCfQLGsEI2dj7JcpL4_30B2dKdtectki")],
    "GEANT-matrix-backbone_network-1_day": [google_drive_url("1g7tmtHfP2k1aaiRTbn4CNztY8Wi-i-fi")],
    "GEANT-matrix-backbone_network-1_hour": [google_drive_url("1DN_a2hILndWxVj4DNXXEYTd_ptWQoVNr")],
    "GEANT-node-backbone_network-15_minutes": [google_drive_url("1Lh47kgtXWnDZKkUNrWsf6y_br3-YoyF7")],
    "GEANT-node-backbone_network-1_day": [google_drive_url("1Sgt5PUq_ncRJzcylus9OjwtymW9NoJNj")],
    "GEANT-node-backbone_network-1_hour": [google_drive_url("1MQp6lg_LZlXLPAAr_ER_vl-LkKAuCIXk")],
    "GEANT-node2node-backbone_network-15_minutes": [google_drive_url("1LEIo6P78Qtrq6fnJjVQgoMvxEENkebrM")],
    "GEANT-node2node-backbone_network-1_day": [google_drive_url("13Ab9dQCz0N6WnQm-5gAyrOZcvUsl8Qn8")],
    "GEANT-node2node-backbone_network-1_hour": [google_drive_url("1kBlAxuW4B7SnAUM9Cm8Mku0DiytF7wrv")],
}

# SHA-256 digests of the dataset files (verified after download).
CHECKSUMS = {
    "GEANT-matrix-backbone_network-15_minutes": "ff8a45e1cb9e6df4a8555be8fce1a80ad7ff35ef83862eb9fd97433b4887ab32",
    "GEANT-matrix-backbone_network-1_day": "9e415fabcea808afe8c6a248857bea411a2d0294e7cf2ce4670db9512c15afb3",
    "GEANT-matrix-backbone_network-1_hour": "cba14524b82413bacb86bd4bf0480b38345e6eebf7e6a22ddb105cc940d1eeac",
    "GEANT-node-backbone_network-15_minutes": "1b987a0cd823aed42cac194dd400b2f4d15f0e1d625a3179cb9c06584b1b01fd",
    "GEANT-node-backbone_network-1_day": "2e7dfc305fce5d70628df259434b1f921253e3a38bd4b9358876e84df6d96d86",
    "GEANT-node-backbone_network-1_hour": "9f8c6aab6e0c241cd37a50c74cd5627da1fbbb97094207106af0ab289203b6a3",
    "GEANT-node2node-backbone_network-15_minutes": "c8195a426958425013dc79a8ee3894298fea4fc263e57da832995712e67c9b0f",
    "GEANT-node2node-backbone_network-1_day": "a406e7496a8cd148abee329e8a89d244470e051735039584f72f6449beb1821e",
    "GEANT-node2node-backbone_network-1_hour": "2980a697962591e02d187316b8bcc326744787e4d5ede107e54b9a9a571eb66a",
}

# Zenodo record https://zenodo.org/records/22721975 (DOI 10.5281/zenodo.22721975, published 2026-10-13) holds the same
# files (byte-identical to the DataZoo bucket, see CHECKSUMS); tried after the Google Drive mirror.
ZENODO_RECORD = 22721975
ZENODO_SOURCES = {name: [zenodo_file_url(ZENODO_RECORD, f"{name}.h5")] for name in CHECKSUMS}
FALLBACK_DOWNLOAD_URLS = {name: list(FALLBACK_DOWNLOAD_URLS.get(name, [])) + ZENODO_SOURCES[name] for name in CHECKSUMS}
