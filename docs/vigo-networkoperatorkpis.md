# Vigo-Network operators KPIs { #vigo_networkoperatorkpis_page }

#### Description
This dataset contains the measurements of different key performance indicators (KPIs) of the usage of a network operator's infrastructure. It provides time series with the evolution of the KPIs measured every 5 minutes for a time interval greather than one month. The measurements correspond to different operator locations. Four different KPIs are provided in the dataset: aggregated Internet traffic (in bits per second), downstream traffic (in bits per second), number of active client sessions, and Virtual Private Network (VPN) traffic (in bits per second).

The results have been anonymized, the time frame has been shifted so that the first timestamp of each time series is 0 and the values of the KPIs have been scaled, so that they range from 0 to 1000 in each time series.

The dataset is divided into four subsets: **Downstream**, **Internet**, **Sessions** and **VPN**. Those subsets are additionaly aggregated into 10-minutes, 1-hour and 1-day time window intervals.

### Downstream subset
In this subset, time series are of downstream traffic. Contains 12 time series.

#### 5-minutes interval time series metrics

| Time Series Metric          | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| id_time                     | Unique identifier for each aggregation interval within the time series, used to segment the dataset. |
| scaled_bts                  | Scaled downstream traffic in bits per second, representing the total downstream throughput for the interval. |

#### 10-minutes, 1-hour and 1-day interval time series metrics

| Time Series Metric           | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| id_time                     | Unique identifier for each aggregation interval within the time series, used to segment the dataset. |
| avg_scaled_bts               | Average downstream traffic over the aggregation interval.                                       |
| std_scaled_bts               | Standard deviation of downstream traffic over the aggregation interval.                         |
| sum_scaled_bts               | Sum of downstream traffic over the aggregation interval.                                       |


### Internet subset
In this subset, time series are of internet traffic. Contains 11 time series.

#### 5-minutes interval time series metrics

| Time Series Metric          | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| id_time                     | Unique identifier for each aggregation interval within the time series, used to segment the dataset. |
| scaled_bts                  | Scaled internet traffic in bits per second, representing the total internet throughput for the interval. |

#### 10-minutes, 1-hour and 1-day interval time series metrics

| Time Series Metric           | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| id_time                     | Unique identifier for each aggregation interval within the time series, used to segment the dataset. |
| avg_scaled_bts               | Average internet traffic over the aggregation interval.                                         |
| std_scaled_bts               | Standard deviation of internet traffic over the aggregation interval.                           |
| sum_scaled_bts               | Sum of internet traffic over the aggregation interval.                                         |

### Sessions subset
In this subset, time series are of number of active client sessions. Contains 21 time series.

#### 5-minutes interval time series metrics

| Time Series Metric                | Description                                                                                     |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| id_time                           | Unique identifier for each aggregation interval within the time series, used to segment the dataset. |
| scaled_active_client_sessions     | Scaled number of active client sessions during the interval, representing concurrent users connected to the network. |

#### 10-minutes, 1-hour and 1-day interval time series metrics

| Time Series Metric                     | Description                                                                                     |
|--------------------------------------|-------------------------------------------------------------------------------------------------|
| id_time                           | Unique identifier for each aggregation interval within the time series, used to segment the dataset. |
| avg_scaled_active_client_sessions     | Average number of active client sessions over the aggregation interval.                          |
| std_scaled_active_client_sessions     | Standard deviation of active client sessions over the aggregation interval.                      |
| sum_scaled_active_client_sessions     | Sum of active client sessions over the aggregation interval.                                     |

### VPN subset
In this subset, time series are of VPN traffic. Contains 11 time series.

#### 5-minutes interval time series metrics

| Time Series Metric          | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| id_time                     | Unique identifier for each aggregation interval within the time series, used to segment the dataset. |
| scaled_bts                  | Scaled VPN traffic in bits per second, representing total VPN throughput for the interval.      |

#### 10-minutes, 1-hour and 1-day interval time series metrics

| Time Series Metric           | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| id_time                     | Unique identifier for each aggregation interval within the time series, used to segment the dataset. |
| avg_scaled_bts               | Average VPN traffic over the aggregation interval.                                             |
| std_scaled_bts               | Standard deviation of VPN traffic over the aggregation interval.                                 |
| sum_scaled_bts               | Sum of VPN traffic over the aggregation interval.                                              |

### Links

More detailed description is available on the [Zenodo](https://doi.org/10.5281/zenodo.8147768). Original dataset can be downloaded at [https://doi.org/10.5281/zenodo.8147768](https://doi.org/10.5281/zenodo.8147768).
