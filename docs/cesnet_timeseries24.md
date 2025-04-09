# CESNET-TimeSeries24 { #cesnet_timeseries24_page }

#### Data capture

The dataset called CESNET-TimeSeries24 was collected by long-term monitoring of selected statistical metrics for 40 weeks for each IP address on the ISP network CESNET3 (Czech Education and Science Network). The dataset encompasses network traffic from more than 275,000 active IP addresses, assigned to a wide variety of devices, including office computers, NATs, servers, WiFi routers, honeypots, and video-game consoles found in dormitories. Moreover, the dataset is also rich in network anomaly types since it contains all types of anomalies, ensuring a comprehensive evaluation of anomaly detection methods.

Last but not least, the CESNET-TimeSeries24 dataset provides traffic time series on institutional and IP subnet levels to cover all possible anomaly detection or forecasting scopes. Overall, the time series dataset was created from the 66 billion IP flows that contain 4 trillion packets that carry approximately 3.7 petabytes of data. The CESNET-TimeSeries24 dataset is a complex real-world dataset that will finally bring insights into the evaluation of forecasting models in real-world environments.

#### Data description

We create evenly spaced time series for each IP address by aggregating IP flow records into time series datapoints. The created datapoints represent the behavior of IP addresses within a defined time window of 10 minutes. The time series are built from multivariate datapoints.  

Datapoints created by the aggregation of IP flows contain the following time-series metrics:

- **Simple volumetric metrics:** the number of IP flows, the number of packets, and the transmitted data size (i.e. number of bytes)
- **Unique volumetric metrics:** the number of unique destination IP addresses, the number of unique destination Autonomous System Numbers (ASNs), and the number of unique destination transport layer ports. The aggregation of *Unique volumetric metrics* is memory intensive since all unique values must be stored in an array. We used a server with 41 GB of RAM, which was enough for 10-minute aggregation on the ISP network.
- **Ratios metrics:** the ratio of UDP/TCP packets, the ratio of UDP/TCP transmitted data size, the direction ratio of packets, and the direction ratio of transmitted data size
- **Average metrics:** the average flow duration, and the average Time To Live (TTL)

#### Multiple time aggregation  

The original datapoints in the dataset are aggregated by 10 minutes of network traffic. The size of the aggregation interval influences anomaly detection procedures, mainly the training speed of the detection model. However, the 10-minute intervals can be too short for longitudinal anomaly detection methods. Therefore, we added two more aggregation intervals to the datasets--1 hour and 1 day.

#### Time series of institutions  

We identify 283 institutions inside the CESNET3 network. These time series aggregated per each institution ID provide a view of the institution's data.

#### Time series of institutional subnets

We identify 548 institution subnets inside the CESNET3 network. These time series aggregated per each institution ID provide a view of the institution subnet's data.

#### ipfixprobe parameters

    Active timeout: 5min
    Inactive timeout: 65s

#### List of time series metrics

The following list describes time series metrics in dataset:

|  Time Series Metric    |   Description                                          |
|------------------------|---------------------------------------------------------------|
| id_time                |  Unique identifier for each aggregation interval within the time series, used to segment the dataset into specific time periods for analysis. |
| n_flows                | Total number of flows observed in the aggregation interval, indicating the volume of distinct sessions or connections for the IP address. |
| n_packets              | Total number of packets transmitted during the aggregation interval, reflecting the packet-level traffic volume for the IP address. |
| n_bytes                | Total number of bytes transmitted during the aggregation interval, representing the data volume for the IP address. |
| n_dest_ip              | Number of unique destination IP addresses contacted by the IP address during the aggregation interval, showing the diversity of endpoints reached. |
| n_dest_asn             | Number of unique destination Autonomous System Numbers (ASNs) contacted by the IP address during the aggregation interval, indicating the diversity of networks reached. |
| n_dest_port            | Number of unique destination transport layer ports contacted by the IP address during the aggregation interval, representing the variety of services accessed. |
| tcp_udp_ratio_packets  | Ratio of packets sent using TCP versus UDP by the IP address during the aggregation interval, providing insight into the transport protocol usage pattern. This metric belongs to the interval <0, 1> where 1 is when all packets are sent over TCP, and 0 is when all packets are sent over UDP. |
| tcp_udp_ratio_bytes    | Ratio of bytes sent using TCP versus UDP by the IP address during the aggregation interval, highlighting the data volume distribution between protocols. This metric belongs to the interval <0, 1>  with same rule as tcp_udp_ratio_packets. |
| dir_ratio_packets      | Ratio of packet directions (inbound versus outbound) for the IP address during the aggregation interval, indicating the balance of traffic flow directions. This metric belongs to the interval <0, 1>, where 1 is when all packets are sent in the outgoing direction from the monitored IP address, and 0 is when all packets are sent in the incoming direction to the monitored IP address. |
| dir_ratio_bytes        | Ratio of byte directions (inbound versus outbound) for the IP address during the aggregation interval, showing the data volume distribution in traffic flows. This metric belongs to the interval <0, 1> with the same rule as dir_ratio_packets. |
| avg_duration           | Average duration of IP flows for the IP address during the aggregation interval, measuring the typical session length. |
| avg_ttl                | Average Time To Live (TTL) of IP flows for the IP address during the aggregation interval, providing insight into the lifespan of packets. |

Moreover, the time series created by re-aggregation contains following time series metrics instead of *n_dest_ip*, *n_dest_asn*, and *n_dest_port*:

|  Time Series Metric    |   Description                                                                |
|------------------------|------------------------------------------------------------------------------|
| sum_n_dest_ip          | Sum of numbers of unique destination IP addresses.                           |
| avg_n_dest_ip          | The average number of unique destination IP addresses.                       |
| std_n_dest_ip          | Standard deviation of numbers of unique destination IP addresses.            |
| sum_n_dest_asn         | Sum of numbers of unique destination ASNs.                                   |
| avg_n_dest_asn         | The average number of unique destination ASNs.                               |
| std_n_dest_asn         | Standard deviation of numbers of unique destination ASNs.                    |
| sum_n_dest_port        | Sum of numbers of unique destination transport layer ports.                  |
| avg_n_dest_port        |  The average number of unique destination transport layer ports.             |
| std_n_dest_port        | Standard deviation of numbers of unique destination transport layer ports.   |

More detailed description is available in the dataset [paper](https://doi.org/10.1038/s41597-025-04603-x) or you can contact dataset author [Josef Koumar](https://koumajos.github.io/).