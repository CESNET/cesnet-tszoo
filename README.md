<p align="center">
    <img src="https://raw.githubusercontent.com/CESNET/cesnet-tszoo/main/docs/images/tszoo.svg" width="450">
</p>

[![](https://img.shields.io/badge/license-BSD-blue.svg)](https://github.com/CESNET/cesnet-tszoo/blob/main/LICENSE)
[![](https://img.shields.io/badge/docs-cesnet--tszoo-blue.svg)](https://cesnet.github.io/cesnet-tszoo/)
[![](https://img.shields.io/badge/python->=3.10-blue.svg)](https://pypi.org/project/cesnet-tszoo/)
[![](https://img.shields.io/pypi/v/cesnet-tszoo)](https://pypi.org/project/cesnet-tszoo/)

The goal of `cesnet-tszoo` project is to provide time series datasets with useful tools for preprocessing and reproducibility. Such as:

- API for downloading, configuring and loading CESNET-TimeSeries24, CESNET-AGG23 datasets. Each with various sources and aggregations.
- Example of configuration options:
  - Data can be split into train/val/test sets. Split can be done by time series or by time periods.
  - Transforming of data with built-in transformers or with custom transformers.
  - Handling missing values built-in fillers or with custom fillers.
  - Applying custom handlers.
  - Changing order of when are preprocesses applied/fitted  
- Creation and import of benchmarks, for easy reproducibility of experiments.
- Creation and import of annotations. Can create annotations for specific time series, specific time or specific time in specific time series.

## Datasets

| Name                      | CESNET-TimeSeries24                                                                       | CESNET-AGG23                                                                                          |
|---------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| _Published in_            | 2025                                                                                      | 2023                                                                                                  |
| _Collection duration_     | 40 weeks                                                                                  | 10 weeks                                                                                              |
| _Collection period_       | 9.10.2023 - 14.7.2024                                                                     | 25.2.2023 - 3.5.2023                                                                                  |
| _Aggregation window_      | 1 day, 1 hour, 10 min                                                                     | 1 min                                                                                                 |
| _Sources_                 | CESNET3: Institutions, Institution subnets, IP addresses                                  | CESNET2                                                                                               |
| _Number of time series_   | Institutions: 849, Institution subnets: 1644, IP addresses: 825372                        | 1                                                                                                     |
| _Cite_                    | [https://doi.org/10.1038/s41597-025-04603-x](https://doi.org/10.1038/s41597-025-04603-x)  | [https://doi.org/10.23919/CNSM59352.2023.10327823](https://doi.org/10.23919/CNSM59352.2023.10327823)  |
| _Zenodo URL_              | [https://zenodo.org/records/13382427](https://zenodo.org/records/13382427)                | [https://zenodo.org/records/8053021](https://zenodo.org/records/8053021)                              |
| _Related papers_          |                                                                                           |                                                                                                       |

## Installation

Install the package from pip with:

```bash
pip install cesnet-tszoo
```

or for editable install with:

```bash
pip install -e git+https://github.com/CESNET/cesnet-tszoo#egg=cesnet-tszoo
```

## Citation

If you use CESNET TS-Zoo, please cite our paper:

```
@misc{kures2025,
    title={CESNET TS-Zoo: A Library for Reproducible Analysis of Network Traffic Time Series}, 
    author={Milan Kureš and Josef Koumar and Karel Hynek},
    booktitle={2025 21th International Conference on Network and Service Management (CNSM)}, 
    year={2025}
}
```

## Examples

For detailed examples refer to [`Tutorial notebooks`](https://github.com/CESNET/cesnet-ts-zoo-tutorials)

### Initialize dataset to create train, validation, and test dataframes

#### Using [`TimeBasedCesnetDataset`](https://cesnet.github.io/cesnet-tszoo/reference_time_based_cesnet_dataset/) dataset

```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType
from cesnet_tszoo.configs import TimeBasedConfig

dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.TIME_BASED)
config = TimeBasedConfig(
    ts_ids=50, # number of randomly selected time series from dataset
    train_time_period=range(0, 100), 
    val_time_period=range(100, 150), 
    test_time_period=range(150, 250), 
    features_to_take=["n_flows", "n_packets"])
dataset.set_dataset_config_and_initialize(config)

train_dataframe = dataset.get_train_df()
val_dataframe = dataset.get_val_df()
test_dataframe = dataset.get_test_df()
```

Time-based datasets are configured with [`TimeBasedConfig`](https://cesnet.github.io/cesnet-tszoo/reference_time_based_config/).

#### Using [`DisjointTimeBasedCesnetDataset`](https://cesnet.github.io/cesnet-tszoo/reference_disjoint_time_based_cesnet_dataset/) dataset
```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType
from cesnet_tszoo.configs import DisjointTimeBasedConfig

dataset = CESNET_TimeSeries24.get_dataset("/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.DISJOINT_TIME_BASED)
config = DisjointTimeBasedConfig(
    train_ts=50, # number of randomly selected time series from dataset that are not in val_ts and test_ts
    val_ts=20, # number of randomly selected time series from dataset that are not in train_ts and test_ts
    test_ts=10, # number of randomly selected time series from dataset that are not in train_ts and val_ts
    train_time_period=range(0, 100), 
    val_time_period=range(100, 150), 
    test_time_period=range(150, 250), 
    features_to_take=["n_flows", "n_packets"])
dataset.set_dataset_config_and_initialize(config)

train_dataframe = dataset.get_train_df()
val_dataframe = dataset.get_val_df()
test_dataframe = dataset.get_test_df()
```

Disjoint-time-based datasets are configured with [`DisjointTimeBasedConfig`](https://cesnet.github.io/cesnet-tszoo/reference_disjoint_time_based_config/).

#### Using [`SeriesBasedCesnetDataset`](https://cesnet.github.io/cesnet-tszoo/reference_series_based_cesnet_dataset/) dataset

```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType
from cesnet_tszoo.configs import SeriesBasedConfig

dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.SERIES_BASED)
config = SeriesBasedConfig(
    time_period=range(0, 250), 
    train_ts=50, # number of randomly selected time series from dataset that are not in val_ts and test_ts
    val_ts=20, # number of randomly selected time series from dataset that are not in train_ts and test_ts
    test_ts=10, # number of randomly selected time series from dataset that are not in train_ts and val_ts
    features_to_take=["n_flows", "n_packets"])
dataset.set_dataset_config_and_initialize(config)

train_dataframe = dataset.get_train_df()
val_dataframe = dataset.get_val_df()
test_dataframe = dataset.get_test_df()
```

Series-based datasets are configured with [`SeriesBasedConfig`](https://cesnet.github.io/cesnet-tszoo/reference_series_based_config/).

#### Using [`load_benchmark`](https://cesnet.github.io/cesnet-tszoo/benchmarks_tutorial/)

```python
from cesnet_tszoo.benchmarks import load_benchmark

benchmark = load_benchmark(identifier="2e92831cb502", data_root="/some_directory/")
dataset = benchmark.get_initialized_dataset()

train_dataframe = dataset.get_train_df()
val_dataframe = dataset.get_val_df()
test_dataframe = dataset.get_test_df()
```

Loaded dataset can be one of the above.
