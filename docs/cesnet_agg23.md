# CESNET-AGG23 { #cesnet_agg23_page }

#### Data capture

The data was captured in the flow monitoring infrastructure of the CESNET2 network. The capturing was done for two months between 25.2.2023 and 3.5.2023.

#### Data description

The dataset consists of rules set used by the Scalar aggregator to aggregate information from incoming flows gathered by ipfixprobe. Each row in the dataset represents a single time 1-minute window interval.

#### ipfixprobe parameters

    Active timeout: 10min
    Inactive timeout: 1min

#### List of time series metrics

|  Time Series Metric    |   Description                                                                                                                                      |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| id_time                   |    Unique identifier for each aggregation interval within the time series, used to segment the dataset into specific time periods for analysis. |
| avr_duration           |    The average duration of all flows.                                                                                                              |
| avr_duration_ipv4      |    The average duration of IPV4 flows.                                                                                                             |
| avr_duration_ipv6      |    The average duration of IPV6 flows.                                                                                                             |
| avr_duration_tcp       |    The average duration of TCP flows.                                                                                                              |
| avr_duration_udp       |    The average duration of UDP flows.                                                                                                              |
| byte_avg               |    The average number of bytes of all flows.                                                                                                       |
| byte_avg_ipv4          |    The average number of bytes of IPV4 flows.                                                                                                      |
| byte_avg_ipv6          |    The average number of bytes of IPV6 flows.                                                                                                      |
| byte_avg_tcp           |    The average number of bytes of TCP flows.                                                                                                       |
| byte_avg_udp           |    The average number of bytes of UDP flows.                                                                                                       |
| byte_rate              |    Byte rate estimation on the network of all flows.                                                                                               |
| byte_rate_ipv4         |    Byte rate estimation on the network of IPV4 flows.                                                                                              |
| byte_rate_ipv6         |    Byte rate estimation on the network of IPV6 flows.                                                                                              |
| byte_rate_tcp          |    Byte rate estimation on the network of TCP flows.                                                                                               |
| byte_rate_udp          |    Byte rate estimation on the network of UDP flows.                                                                                               |
| bytes                  |    The sum of bytes of all flows.                                                                                                                  |
| bytes_ipv4             |    The sum of bytes of IPV4 flows.                                                                                                                 |
| bytes_ipv6             |    The sum of bytes of IPV6 flows.                                                                                                                 |
| bytes_tcp              |    The sum of bytes of TCP flows.                                                                                                                  |
| bytes_udp              |    The sum of bytes of UDP flows.                                                                                                                  |
| no_flows               |    The number of all active flows.                                                                                                                 |
| no_flows_ipv4          |    The number of IPV4 active flows.                                                                                                                |
| no_flows_ipv6          |    The number of IPV6 active flows.                                                                                                                |
| no_flows_tcp           |    The number of TCP active flows.                                                                                                                 |
| no_flows_tcp_synonly   |    The number of flows containing only SYN packets.                                                                                                |
| no_flows_udp           |    The number of UDP active flows.                                                                                                                 |
| no_uniq_biflows        |    The number of all unique biflows.                                                                                                               |
| no_uniq_flows          |    The number of all unique flows.                                                                                                                 |
| packet_avg             |    The average packets of all flows.                                                                                                               |
| packet_avg_ipv4        |    The average packets of IPV4 flows.                                                                                                              |
| packet_avg_ipv6        |    The average packets of IPV6 flows.                                                                                                              |
| packet_avg_tcp         |    The average packets of TCP flows.                                                                                                               |
| packet_avg_udp         |    The average packets of UDP flows.                                                                                                               |
| packet_rate            |    Packet rate estimation on the network of all flows.                                                                                             |
| packet_rate_ipv4       |    Packet rate estimation on the network of IPV4 flows.                                                                                            |
| packet_rate_ipv6       |    Packet rate estimation on the network of IPV6 flows.                                                                                            |
| packet_rate_tcp        |    Packet rate estimation on the network of TCP flows.                                                                                             |
| packet_rate_udp        |    Packet rate estimation on the network of UDP flows.                                                                                             |
| packets                |    The sum of packets of all flows.                                                                                                                |
| packets_ipv4           |    The sum of packets of IPV4 flows.                                                                                                               |
| packets_ipv6           |    The sum of packets of IPV6 flows.                                                                                                               |
| packets_tcp            |    The sum of packets of TCP flows.                                                                                                                |
| packets_udp            |    The sum of packets of UDP flows.                                                                                                                |

More detailed description is available in the [paper](https://doi.org/10.23919/CNSM59352.2023.10327823) or you can contact dataset author [Jaroslav Pesek](https://jaroslavpesek.github.io/).
