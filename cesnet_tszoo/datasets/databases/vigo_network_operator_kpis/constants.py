import numpy as np

from cesnet_tszoo.utils.download import google_drive_url, zenodo_file_url
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

# Secondary sources tried when the DataZoo bucket is unreachable; the files are copies of the bucket content.
FALLBACK_DOWNLOAD_URLS = {
    "Vigo-NetworkOperatorKPIs-downstream-network_operator-10_minutes": [google_drive_url("16bAhUIDCLV4T3PUED_-tIlvqppQdgyt6")],
    "Vigo-NetworkOperatorKPIs-downstream-network_operator-1_day": [google_drive_url("1NRYejgIy9f5qeYtYUGNgnVh7PVDqz_HI")],
    "Vigo-NetworkOperatorKPIs-downstream-network_operator-1_hour": [google_drive_url("1wKaYM2IMWasRTm-YuCEDKyp5pQLMkyy6")],
    "Vigo-NetworkOperatorKPIs-downstream-network_operator-5_minutes": [google_drive_url("1K7lBEeB4rO_4XCrPbtm_7JMNHg95I4Sc")],
    "Vigo-NetworkOperatorKPIs-internet-network_operator-10_minutes": [google_drive_url("1uIKwYn2cIRaOlkgB0ypOBjyNWQ6aBYAA")],
    "Vigo-NetworkOperatorKPIs-internet-network_operator-1_day": [google_drive_url("1OSSUn4n92zWfgOj2u_kK7ZemNUPBEuS3")],
    "Vigo-NetworkOperatorKPIs-internet-network_operator-1_hour": [google_drive_url("1LueaKSOClY383epUWdqKYvARgbKWATvm")],
    "Vigo-NetworkOperatorKPIs-internet-network_operator-5_minutes": [google_drive_url("17OhRymU_NsGZlwymgc31ZA_3DcNQFwAH")],
    "Vigo-NetworkOperatorKPIs-sessions-network_operator-10_minutes": [google_drive_url("1YTQY0V9qq_Z6F1SW3Q5vm_BpCZivbqq-")],
    "Vigo-NetworkOperatorKPIs-sessions-network_operator-1_day": [google_drive_url("12EKFGtGQPc1L60TItJdFdo9KNAsCBOYm")],
    "Vigo-NetworkOperatorKPIs-sessions-network_operator-1_hour": [google_drive_url("1-q3rcwzVs7IPYRcfsn426s75I-Tr-kCB")],
    "Vigo-NetworkOperatorKPIs-sessions-network_operator-5_minutes": [google_drive_url("1uGKXOhdFvqP-uRKK6ztvfrujNfaNwq33")],
    "Vigo-NetworkOperatorKPIs-vpn-network_operator-10_minutes": [google_drive_url("1mNwyP2KoC7goMU-_T6Aqc4aGLvzydRCm")],
    "Vigo-NetworkOperatorKPIs-vpn-network_operator-1_day": [google_drive_url("1n1HQUxCgbtB3NFZIMjsDGi0mDALj9v2c")],
    "Vigo-NetworkOperatorKPIs-vpn-network_operator-1_hour": [google_drive_url("1Aq6iWvCeXuF2RW3VXiq7-fvD4cdIWaxI")],
    "Vigo-NetworkOperatorKPIs-vpn-network_operator-5_minutes": [google_drive_url("1eQvoSbg28an7enO1fmKZg3f-pJ4FV-is")],
}

# SHA-256 digests of the dataset files (verified after download).
CHECKSUMS = {
    "Vigo-NetworkOperatorKPIs-downstream-network_operator-10_minutes": "6a502065fd0d2d7c680855e72d5b8dc59ed97d438dee805081bb531b90728b83",
    "Vigo-NetworkOperatorKPIs-downstream-network_operator-1_day": "bb480e7ea55bd335036c524d55e52d7e24b7810baff880152696f318e63c1b45",
    "Vigo-NetworkOperatorKPIs-downstream-network_operator-1_hour": "bc489116d990c2122858e82b0d5cded6ae4feb20f8af408f929965d4e01d652b",
    "Vigo-NetworkOperatorKPIs-downstream-network_operator-5_minutes": "56f44fed88d447353189ec56d137a4592f65a932ef8a684d7135190a156591cd",
    "Vigo-NetworkOperatorKPIs-internet-network_operator-10_minutes": "11a0e19e469fc40438dfaae58cbfe0f81549cdab8cb1b137f543216f121e1529",
    "Vigo-NetworkOperatorKPIs-internet-network_operator-1_day": "e57352a0e4bfbed7e41f578cf9d6a3e752ce47b8991ddd1db791a5cfa2424690",
    "Vigo-NetworkOperatorKPIs-internet-network_operator-1_hour": "d9c4ebf34323635373547da6a882165a155a4b29db5ad137192d641ec64a41ab",
    "Vigo-NetworkOperatorKPIs-internet-network_operator-5_minutes": "ea733dc10df6ebcea966c0822bb9a0d1366643828cc72ac67b61ec154c423f1b",
    "Vigo-NetworkOperatorKPIs-sessions-network_operator-10_minutes": "094641570ae163284b344e86fe541d999962d7379ea715aa7fc3b5573dbc7f5a",
    "Vigo-NetworkOperatorKPIs-sessions-network_operator-1_day": "b5c0243955d0cf7bab80738dbcf835ca7f5b4a2dd1a4cf3e88d1c22b2d116cdc",
    "Vigo-NetworkOperatorKPIs-sessions-network_operator-1_hour": "0b31cbc4cc6abd554057fa7ffc5980acbc168ae315f8de8dac9271091275a31e",
    "Vigo-NetworkOperatorKPIs-sessions-network_operator-5_minutes": "b36fc97c21a89acb2e658eb8e5d83e1a639fe4df84f264543595c8f5778be50a",
    "Vigo-NetworkOperatorKPIs-vpn-network_operator-10_minutes": "b3714003569929a486ae7d6cf47819f41fb500000552bf266d6a12ac035f7e85",
    "Vigo-NetworkOperatorKPIs-vpn-network_operator-1_day": "a761a0620f5059e2c7cc77dd5a554cc31f472698234af5f71f66f82b7475d2e8",
    "Vigo-NetworkOperatorKPIs-vpn-network_operator-1_hour": "766c5876c7f3abf662488cf8f31deddeeb7f73002d3eebf439a74cfb03b0c3b8",
    "Vigo-NetworkOperatorKPIs-vpn-network_operator-5_minutes": "bc75f5184603eba242b25677410dc98bd26f17877e629fcd01dde40c02b0f991",
}

# Zenodo record https://zenodo.org/records/22721975 (DOI 10.5281/zenodo.22721975, published 2026-10-13) holds the same
# files (byte-identical to the DataZoo bucket, see CHECKSUMS); tried after the Google Drive mirror.
ZENODO_RECORD = 22721975
ZENODO_SOURCES = {name: [zenodo_file_url(ZENODO_RECORD, f"{name}.h5")] for name in CHECKSUMS}
FALLBACK_DOWNLOAD_URLS = {name: list(FALLBACK_DOWNLOAD_URLS.get(name, [])) + ZENODO_SOURCES[name] for name in CHECKSUMS}
