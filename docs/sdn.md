# SDN { #sdn_page }

#### Data capture
The dataset was generated in a software‑defined network (SDN) emulation environment built using the Mininet simulator and managed by a POX SDN controller. Traffic matrix measurements were obtained by periodically polling flow statistics from OpenFlow switches deployed in a simulated network topology based on the BSO Network 2011 topology. Hosts in the SDN testbed replayed modified traffic capture (pcap) files to generate controlled traffic flows between all pairs of nodes. Traffic matrices were recorded at regular intervals over a multi‑day test run, producing thousands of measured matrices.

#### Data description
The dataset consists of traffic matrices capturing the volume of network traffic between origin–destination (OD) node pairs in the SDN testbed at each measurement interval. The created datapoints represent a summarized traffic volumes between origin–destination node pairs within a defined time window of 1 minute.

The dataset is divided into three subsets: **Matrix**, **Node2Node**, and **Node**. Each subset is additionaly aggregated into 10-minutes, 1-hour and 1-day time window intervals.

### Matrix subset
In this subset, features are 14x14 TMs. Contains 1 time series.

#### 1-minute interval time series metrics

| Time Series Metric          | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| matrix_traffic_volume       | Traffic volume between all origin–destination node pairs in the SDN testbed network at each time interval. |

#### 10-minutes, 1-hour and 1-day interval time series metrics

| Time Series Metric           | Description                                                                                                         |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_matrix_traffic_volume    | Average traffic volume between all OD node pairs over the aggregation interval.                                    |
| std_matrix_traffic_volume    | Standard deviation of traffic volume between all OD node pairs over the aggregation interval.                     |
| sum_matrix_traffic_volume    | Sum of traffic volume between all OD node pairs over the aggregation interval.                                     |

### Node2Node subset
In this subset, each time series is a cell from the TMs. Contains 196 time series.

#### 1-minute interval time series metrics

| Time Series Metric          | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| matrix_traffic_volume       | Traffic volume between a specific source–destination node pair in the SDN testbed at each time interval. |

#### 10-minutes, 1-hour and 1-day interval time series metrics

| Time Series Metric           | Description                                                                                                         |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_matrix_traffic_volume    | Average traffic volume of a specific OD node pair over the aggregation interval.                                   |
| std_matrix_traffic_volume    | Standard deviation of traffic volume of a specific OD node pair over the aggregation interval.                     |
| sum_matrix_traffic_volume    | Sum of traffic volume of a specific OD node pair over the aggregation interval.                                   |

### Node subset
In this subset, each time series corresponds to one node. Contains 14 time series.

#### Time series metrics

| Time Series Metric           | Description                                                                                                         |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_from_traffic_volume      | Average traffic volume originating from the node over the aggregation interval.    |
| std_from_traffic_volume      | Standard deviation of traffic volume originating from the node over the aggregation interval. |
| sum_from_traffic_volume      | Sum of traffic volume originating from the node over the aggregation interval.      |
| avg_to_traffic_volume        | Average traffic volume destined to the node over the aggregation interval.        |
| std_to_traffic_volume        | Standard deviation of traffic volume destined to the node over the aggregation interval. |
| sum_to_traffic_volume        | Sum of traffic volume destined to the node over the aggregation interval.          |

More detailed description is available in the [paper](https://ieeexplore.ieee.org/document/9500331).
