from cesnet_tszoo.utils.download import google_drive_url, zenodo_file_url
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

# Secondary sources tried when the DataZoo bucket is unreachable; the files are copies of the bucket content.
FALLBACK_DOWNLOAD_URLS = {
    "TelecomItalia-telecommunications-service_provider-10_minutes": [google_drive_url("1Mplk3oF6CSD59zFiAOAzEQjZq_k5oTF4")],
    "TelecomItalia-telecommunications-service_provider-1_day": [google_drive_url("1LpKiYWeO30MPAhaP8gL6FM8wAqrfnw4L")],
    "TelecomItalia-telecommunications-service_provider-1_hour": [google_drive_url("1b-6NW_QW3XvYcs4a8WkRyBDYjjpou6xG")],
}

# SHA-256 digests of the dataset files (verified after download).
CHECKSUMS = {
    "TelecomItalia-telecommunications-service_provider-10_minutes": "c40062b8a1f0195fb22c9d8cc1b9dddbdb7232f5c645c5ba2171abc8c4f72e7e",
    "TelecomItalia-telecommunications-service_provider-1_day": "0c3364e1735622543014881573231c1603b5c92ce1afee4700db899007215705",
    "TelecomItalia-telecommunications-service_provider-1_hour": "59703f46b395ef46d5722ed02db6db8f650bf4e8a78afc312534a58c18653135",
}

# Zenodo record https://zenodo.org/records/22721975 (DOI 10.5281/zenodo.22721975, published 2026-10-13) holds the same
# files (byte-identical to the DataZoo bucket, see CHECKSUMS); tried after the Google Drive mirror.
ZENODO_RECORD = 22721975
ZENODO_SOURCES = {name: [zenodo_file_url(ZENODO_RECORD, f"{name}.h5")] for name in CHECKSUMS}
FALLBACK_DOWNLOAD_URLS = {name: list(FALLBACK_DOWNLOAD_URLS.get(name, [])) + ZENODO_SOURCES[name] for name in CHECKSUMS}
