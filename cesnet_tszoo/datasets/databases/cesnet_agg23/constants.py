from cesnet_tszoo.utils.download import google_drive_url, zenodo_file_url
from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.CESNET2: "id"
}

DEFAULT_VALUES = {
    'avr_duration': 0,
    'avr_duration_ipv4': 0,
    'avr_duration_ipv6': 0,
    'avr_duration_tcp': 0,
    'avr_duration_udp': 0,
    'byte_avg': 0,
    'byte_avg_ipv4': 0,
    'byte_avg_ipv6': 0,
    'byte_avg_tcp': 0,
    'byte_avg_udp': 0,
    'byte_rate': 0,
    'byte_rate_ipv4': 0,
    'byte_rate_ipv6': 0,
    'byte_rate_tcp': 0,
    'byte_rate_udp': 0,
    'bytes': 0,
    'bytes_ipv4': 0,
    'bytes_ipv6': 0,
    'bytes_tcp': 0,
    'bytes_udp': 0,
    'no_flows': 0,
    'no_flows_ipv4': 0,
    'no_flows_ipv6': 0,
    'no_flows_tcp': 0,
    'no_flows_tcp_synonly': 0,
    'no_flows_udp': 0,
    'no_uniq_biflows': 0,
    'no_uniq_flows': 0,
    'packet_avg': 0,
    'packet_avg_ipv4': 0,
    'packet_avg_ipv6': 0,
    'packet_avg_tcp': 0,
    'packet_avg_udp': 0,
    'packet_rate': 0,
    'packet_rate_ipv4': 0,
    'packet_rate_ipv6': 0,
    'packet_rate_tcp': 0,
    'packet_rate_udp': 0,
    'packets': 0,
    'packets_ipv4': 0,
    'packets_ipv6': 0,
    'packets_tcp': 0,
    'packets_udp': 0,
}

SOURCE_TYPES = {
    SourceType.CESNET2
}

AGGREGATIONS = {
    AgreggationType.AGG_1_MINUTE
}

# Secondary sources tried when the DataZoo bucket is unreachable; the files are copies of the bucket content.
FALLBACK_DOWNLOAD_URLS = {
    "CESNET-AGG23-CESNET2-minute": [google_drive_url("1we-tL84fHavzwx0uWSvdeERr6lqh9LH7")],
}

# SHA-256 digests of the dataset files (verified after download).
CHECKSUMS = {
    "CESNET-AGG23-CESNET2-minute": "23c6f9720f3cba4ec84f18054822e4eb21e43346037a0b5a13e0f8625805611b",
}

# Zenodo record https://zenodo.org/records/22721975 (DOI 10.5281/zenodo.22721975, published 2026-10-13) holds the same
# files (byte-identical to the DataZoo bucket, see CHECKSUMS); tried after the Google Drive mirror.
ZENODO_RECORD = 22721975
ZENODO_SOURCES = {name: [zenodo_file_url(ZENODO_RECORD, f"{name}.h5")] for name in CHECKSUMS}
FALLBACK_DOWNLOAD_URLS = {name: list(FALLBACK_DOWNLOAD_URLS.get(name, [])) + ZENODO_SOURCES[name] for name in CHECKSUMS}
