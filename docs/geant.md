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

<!-- terms:start -->
### Source, citation and terms of use

- **Original data:** GEANT intradomain traffic matrices, TOTEM project (UCLouvain) — <https://totem.info.ucl.ac.be/dataset.html>
- **Reference:** S. Uhlig, B. Quoitin, J. Lepropre and S. Balon, "Providing public intradomain traffic matrices to the research community," ACM SIGCOMM Computer Communication Review, vol. 36, no. 1, pp. 83-86, January 2006.
- **Licence of the original data:** No licence granted by the source; citation required

If you use this dataset, cite the original work:

```bibtex
@article{uhlig2006providing,
  author = {Uhlig, Steve and Quoitin, Bruno and Lepropre, Jean and Balon, Simon},
  title = {Providing public intradomain traffic matrices to the research community},
  journal = {ACM SIGCOMM Computer Communication Review},
  volume = {36},
  number = {1},
  pages = {83--86},
  year = {2006},
  doi = {10.1145/1111322.1111341}
}
```

!!! warning "Terms of use"
    The files are a converted copy of the original data, redistributed **only for non-commercial research and education**. **Do not use them for commercial purposes.** All rights to the data remain with the original authors and the terms of the original source apply. For any other use, obtain the data and the permission from the original source. Any publication using the data must cite the TOTEM paper above. The conversion to HDF5 does not alter the measured values and is provided as is, without warranty. Rights holders who object to the redistribution can contact the maintainers and the files will be removed.

<!-- terms:end -->

### Links

More detailed description is available in the [paper](https://dl.acm.org/doi/10.1145/1111322.1111341). Original dataset can be downloaded at [https://totem.info.ucl.ac.be/dataset.html](https://totem.info.ucl.ac.be/dataset.html).
