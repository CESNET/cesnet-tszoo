# GÉANT { #geant_page }

#### Data capture

The data was collected from the GÉANT backbone network, a pan‑European research and education network connecting national research and education networks across Europe. It includes measurements taken over several months at regular 15‑minute intervals, where traffic matrices were constructed using sampled traffic data, routing information (IGP/BGP), and other network measurements.

#### Data description
The dataset consists of aggregated traffic matrices representing communication between backbone nodes of the GÉANT network. The datapoints capture summarized traffic volumes between origin–destination node pairs within a defined time window of 15 minutes. The dataset reflects typical backbone traffic patterns over several months and includes anonymized node and link information.

The dataset is divided into three subsets: **Matrix**, **Node2Node**, and **Node**. Each subset is additionaly aggregated into 1-hour and 1-day time window intervals.

### Matrix subset
In this subset, features are 23x23 TMs. Contains 1 time series.

#### 15-minutes interval time series metrics

| Time Series Metric        | Description                                                                                                         |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| matrix_bandwidth_kbs      | The bandwidth of traffic (in kilobits per second) between a source‑destination node pair in the GEANT network at each time interval. |

#### 1-hour and 1-day interval time series metrics

| Time Series Metric           | Description                                                                                                         |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_matrix_bandwidth_kbs     | Average bandwidth (in kilobits per second) between a source‑destination node pair over the aggregation interval.    |
| std_matrix_bandwidth_kbs     | Standard deviation of bandwidth (in kilobits per second) between a source‑destination node pair over the aggregation interval. |
| sum_matrix_bandwidth_kbs     | Sum of bandwidth (in kilobits per second) between a source‑destination node pair over the aggregation interval.     |

### Node2Node subset
In this subset, each time series is a cell from the TMs. Contains 529 time series.

#### 15-minutes interval time series metrics

| Time Series Metric          | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| matrix_bandwidth_kbs        | Bandwidth (in kilobits per second) between a specific source–destination node pair in the GEANT network at each time interval. |

#### 1-hour and 1-day interval time series metrics

| Time Series Metric           | Description                                                                                                         |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_matrix_bandwidth_kbs     | Average bandwidth (in kilobits per second) of a specific source–destination node pair over the aggregation interval. |
| std_matrix_bandwidth_kbs     | Standard deviation of bandwidth (in kilobits per second) of a specific source–destination node pair over the aggregation interval. |
| sum_matrix_bandwidth_kbs     | Sum of bandwidth (in kilobits per second) of a specific source–destination node pair over the aggregation interval.     |

### Node subset
In this subset, each time series corresponds to one node. Contains 23 time series.

#### Time series metrics

| Time Series Metric           | Description                                                                                                         |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_from_bandwidth_kbs       | Average bandwidth (in kilobits per second) originating from the node over the aggregation interval. |
| std_from_bandwidth_kbs       | Standard deviation of bandwidth (in kilobits per second) originating from the node over the aggregation interval. |
| sum_from_bandwidth_kbs       | Sum of bandwidth (in kilobits per second) originating from the node over the aggregation interval. |
| avg_to_bandwidth_kbs         | Average bandwidth (in kilobits per second) destined to the node over the aggregation interval. |
| std_to_bandwidth_kbs         | Standard deviation of bandwidth (in kilobits per second) destined to the node over the aggregation interval. |
| sum_to_bandwidth_kbs         | Sum of bandwidth (in kilobits per second) destined to the node over the aggregation interval. |

More detailed description is available in the [paper](https://dl.acm.org/doi/10.1145/1111322.1111341).
