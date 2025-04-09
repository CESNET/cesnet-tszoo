from cesnet_tszoo.utils.enums import SourceType, AgreggationType
from datetime import datetime
import numpy as np

# CESNET-TimeSeries24
_CESNET_TIME_SERIES24_ID_NAMES = {
    SourceType.IP_ADDRESSES_FULL: "id_ip",
    SourceType.IP_ADDRESSES_SAMPLE: "id_ip",
    SourceType.INSTITUTION_SUBNETS: "id_institution_subnet",
    SourceType.INSTITUTIONS: "id_institution"
}

_CESNET_TIME_SERIES24_DEFAULT_VALUES = {
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

_CESNET_TIME_SERIES24_SOURCE_TYPES = {
    SourceType.INSTITUTION_SUBNETS,
    SourceType.INSTITUTIONS,
    SourceType.IP_ADDRESSES_FULL,
    SourceType.IP_ADDRESSES_SAMPLE
}

_CESNET_TIME_SERIES24_AGGREGATIONS = {
    AgreggationType.AGG_10_MINUTES,
    AgreggationType.AGG_1_HOUR,
    AgreggationType.AGG_1_DAY
}

_CESNET_TIME_SERIES24_ADDITIONAL_DATA = {
    "ids_relationship": (("id_ip", np.int64), ("id_institution", np.int64), ("id_institution_subnet", np.int64)),
    "weekends_and_holidays": (("Date", datetime), ("Type", str))
}

# CESNET-AGG23
_CESNET_AGG23_ID_NAMES = {
    SourceType.CESNET2: "id"
}

_CESNET_AGG23_DEFAULT_VALUES = {
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

_CESNET_AGG23_SOURCE_TYPES = {
    SourceType.CESNET2
}

_CESNET_AGG23_AGGREGATIONS = {
    AgreggationType.AGG_1_MINUTE
}
