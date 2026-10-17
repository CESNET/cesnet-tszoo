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

<!-- terms:start -->
### Source, citation and terms of use

- **Original data:** CESNET-AGG23 (Zenodo record 8053021) — <https://zenodo.org/records/8053021>
- **Reference:** T. Benes, J. Pesek, T. Cejka, "Look at my Network: An Insight into the ISP Backbone Traffic", in Proc. 2023 19th International Conference on Network and Service Management (CNSM), IEEE, 2023, pp. 1-7, https://doi.org/10.23919/CNSM59352.2023.10327823. Dataset: T. Benes, J. Pesek, T. Cejka, "CESNET-AGG23", Zenodo, 18 Jun 2023, https://doi.org/10.5281/zenodo.8053021.
- **Licence of the original data:** CC BY 4.0

If you use this dataset, cite the original work:

```bibtex
@inproceedings{Benes2023LookAtMyNetwork, author={Tomas Benes and Jaroslav Pesek and Tomas Cejka}, title={Look at my Network: An Insight into the {ISP} Backbone Traffic}, booktitle={2023 19th International Conference on Network and Service Management (CNSM)}, pages={1--7}, year={2023}, publisher={IEEE}, doi={10.23919/CNSM59352.2023.10327823}}
@dataset{Benes2023AGG23, author={Benes, Tomas and Pesek, Jaroslav and Cejka, Tomas}, title={{CESNET-AGG23}}, publisher={Zenodo}, year={2023}, month={jun}, doi={10.5281/zenodo.8053021}}
```

!!! warning "Terms of use"
    The files are a converted copy of the original data, distributed under the licence of the original data (CC BY 4.0) and intended for research and education. The terms of the original source apply; check them before any commercial use. The conversion to HDF5 does not alter the measured values and is provided as is, without warranty. Rights holders who object to the redistribution can contact the maintainers and the files will be removed.

<!-- terms:end -->

### Links

More detailed description is available in the [paper](https://doi.org/10.23919/CNSM59352.2023.10327823) or you can contact dataset author [Jaroslav Pesek](https://jaroslavpesek.github.io/). Original dataset can be downloaded at [https://zenodo.org/records/8053021](https://zenodo.org/records/8053021).
