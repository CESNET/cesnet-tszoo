# SDN { #sdn_page }

#### Data capture
The dataset was generated in a software‑defined network (SDN) emulation environment built using the Mininet simulator and managed by a POX SDN controller. Traffic matrix measurements were obtained by periodically polling flow statistics from OpenFlow switches deployed in a simulated network topology based on the BSO Network 2011 topology. Hosts in the SDN testbed replayed modified traffic capture (pcap) files to generate controlled traffic flows between all pairs of nodes. Traffic matrices were recorded at regular intervals over a 4‑day test run, producing thousands of measured matrices.

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

<!-- terms:start -->
### Source, citation and terms of use

- **Original data:** SDN testbed traffic matrices (dataset/testbed_flat_tms.csv) in duchuyle108/SDN-TMprediction — <https://github.com/duchuyle108/SDN-TMprediction>
- **Reference:** D.-H. Le, H.-A. Tran, S. Souihi and A. Mellouk, "An AI-based traffic matrix prediction solution for software-defined network," in ICC 2021 - IEEE International Conference on Communications, 2021, pp. 1-6.
- **Licence of the original data:** No licence stated by the source

If you use this dataset, cite the original work:

```bibtex
@inproceedings{le2021aibased,
  author = {Le, Duc-Huy and Tran, Hai-Anh and Souihi, Sami and Mellouk, Abdelhamid},
  title = {An {AI}-based Traffic Matrix Prediction Solution for Software-Defined Network},
  booktitle = {ICC 2021 - IEEE International Conference on Communications},
  pages = {1--6},
  year = {2021},
  publisher = {IEEE},
  doi = {10.1109/ICC42927.2021.9500331}
}
```

!!! warning "Terms of use"
    The files are a converted copy of the original data, redistributed **only for non-commercial research and education**. **Do not use them for commercial purposes.** All rights to the data remain with the original authors and the terms of the original source apply. For any other use, obtain the data and the permission from the original source. The conversion to HDF5 does not alter the measured values and is provided as is, without warranty. Rights holders who object to the redistribution can contact the maintainers and the files will be removed.

<!-- terms:end -->

### Links

More detailed description is available in the [paper](https://doi.org/10.1109/ICC42927.2021.9500331). Original dataset can be downloaded at [https://github.com/duchuyle108/SDN-TMprediction](https://github.com/duchuyle108/SDN-TMprediction/blob/main/dataset/testbed_flat_tms.csv).
