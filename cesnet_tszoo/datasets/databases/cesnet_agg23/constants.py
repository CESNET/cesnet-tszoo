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
