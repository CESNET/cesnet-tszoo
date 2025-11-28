from datetime import datetime

import numpy as np

from cesnet_tszoo.utils.enums import SourceType, AgreggationType

ID_NAMES = {
    SourceType.IP_ADDRESSES_FULL: "id_ip",
    SourceType.IP_ADDRESSES_SAMPLE: "id_ip",
    SourceType.INSTITUTION_SUBNETS: "id_institution_subnet",
    SourceType.INSTITUTIONS: "id_institution"
}

DEFAULT_VALUES = {
    'n_flows': 0,
    'n_packets': 0,
    'n_bytes': 0,
    'n_dest_ip': 0,
    'n_dest_asn': 0,
    'n_dest_ports': 0,
    'tcp_udp_ratio_packets': 0.5,
    'tcp_udp_ratio_bytes': 0.5,
    'dir_ratio_packets': 0.5,
    'dir_ratio_bytes': 0.5,
    'avg_duration': 0,
    'avg_ttl': 0,
    'sum_n_dest_asn': 0,
    'avg_n_dest_asn': 0,
    'std_n_dest_asn': 0,
    'sum_n_dest_ports': 0,
    'avg_n_dest_ports': 0,
    'std_n_dest_ports': 0,
    'sum_n_dest_ip': 0,
    'avg_n_dest_ip': 0,
    'std_n_dest_ip': 0,
}

SOURCE_TYPES = {
    SourceType.INSTITUTION_SUBNETS,
    SourceType.INSTITUTIONS,
    SourceType.IP_ADDRESSES_FULL,
    SourceType.IP_ADDRESSES_SAMPLE
}

AGGREGATIONS = {
    AgreggationType.AGG_10_MINUTES,
    AgreggationType.AGG_1_HOUR,
    AgreggationType.AGG_1_DAY
}

ADDITIONAL_DATA = {
    "ids_relationship": (("id_ip", np.int64), ("id_institution", np.int64), ("id_institution_subnet", np.int64)),
    "weekends_and_holidays": (("Date", datetime), ("Type", str))
}
