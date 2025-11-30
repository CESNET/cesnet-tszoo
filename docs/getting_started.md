# Getting started

!!! info "Note"
    For a demonstration of usage for simple forecasting refer to Jupyter notebook [`simple_forecasting`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/simple_forecasting.ipynb)

## Code snippets

### Download a dataset
```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType

dataset = CESNET_TimeSeries24.get_dataset("/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.TIME_BASED)
```

Alternatively you can use [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark].

```python
from cesnet_tszoo.benchmarks import load_benchmark

benchmark = load_benchmark("SOME_BUILT_IN_IDENTIFIER", "/some_directory/")
dataset = benchmark.get_dataset()
```

This will create following directories:

- "/some_directory/tszoo"
    - "/some_directory/tszoo/annotations"
    - "/some_directory/tszoo/benchmarks"
    - "/some_directory/tszoo/configs"
    - "/some_directory/tszoo/databases"

Dataset will be downloaded to "/some_directory/tszoo/databases/CESNET_TimeSeries24/".

### Enable logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
```
Set up logging to get more information from the package.

### Initialize dataset to create train, validation, and test sets

#### Using [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset) dataset
```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType
from cesnet_tszoo.configs import TimeBasedConfig

dataset = CESNET_TimeSeries24.get_dataset("/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.TIME_BASED)
config = TimeBasedConfig(
    ts_ids=50, # number of randomly selected time series from dataset
    train_time_period=range(0, 100), 
    val_time_period=range(100, 150), 
    test_time_period=range(150, 250), 
    features_to_take=["n_flows", "n_packets"])
dataset.set_dataset_config_and_initialize(config)
```
Time-based datasets are configured with [`TimeBasedConfig`](reference_time_based_config.md#references.TimeBasedConfig).
Can load data using:

- [`get_train_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`get_val_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader), [`get_test_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader), [`get_all_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_all_dataloader)

- [`get_train_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df), [`get_val_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df), [`get_test_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df), [`get_all_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_all_df)

- [`get_train_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy), [`get_val_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy), [`get_test_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy), [`get_all_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_all_numpy)

#### Using [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset) dataset
```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType
from cesnet_tszoo.configs import DisjointTimeBasedConfig

dataset = CESNET_TimeSeries24.get_dataset("/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.DISJOINT_TIME_BASED)
config = TimeBasedConfig(
    train_ts=50, # number of randomly selected time series from dataset that are not in val_ts and test_ts
    val_ts=20, # number of randomly selected time series from dataset that are not in train_ts and test_ts
    test_ts=10, # number of randomly selected time series from dataset that are not in train_ts and val_ts
    train_time_period=range(0, 100), 
    val_time_period=range(100, 150), 
    test_time_period=range(150, 250), 
    features_to_take=["n_flows", "n_packets"])
dataset.set_dataset_config_and_initialize(config)
```
Disjoint-time-based datasets are configured with [`DisjointTimeBasedConfig`](reference_disjoint_time_based_config.md#references.DisjointTimeBasedConfig).
Can load data using:

- [`get_train_dataloader`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_dataloader), [`get_val_dataloader`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_dataloader), [`get_test_dataloader`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_dataloader)

- [`get_train_df`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_df), [`get_val_df`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_df), [`get_test_df`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_df)

- [`get_train_numpy`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_numpy), [`get_val_numpy`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_numpy), [`get_test_numpy`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_numpy)

#### Using [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) dataset
```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType
from cesnet_tszoo.configs import SeriesBasedConfig

dataset = CESNET_TimeSeries24.get_dataset("/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.SERIES_BASED)
config = SeriesBasedConfig(
    time_period=range(0, 250), 
    train_ts=50, # number of randomly selected time series from dataset that are not in val_ts and test_ts
    val_ts=20, # number of randomly selected time series from dataset that are not in train_ts and test_ts
    test_ts=10, # number of randomly selected time series from dataset that are not in train_ts and val_ts
    features_to_take=["n_flows", "n_packets"])
dataset.set_dataset_config_and_initialize(config)
```
Series-based datasets are configured with [`SeriesBasedConfig`](reference_series_based_config.md#references.SeriesBasedConfig).
Can load data using:

- [`get_train_dataloader`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_dataloader), [`get_val_dataloader`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_dataloader), [`get_test_dataloader`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_dataloader), [`get_all_dataloader`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_all_dataloader)

- [`get_train_df`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_df), [`get_val_df`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_df), [`get_test_df`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_df), [`get_all_df`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_all_df)

- [`get_train_numpy`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_numpy), [`get_val_numpy`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_numpy), [`get_test_numpy`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_numpy), [`get_all_numpy`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_all_numpy)

#### Using [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark]
```python
from cesnet_tszoo.benchmarks import load_benchmark

benchmark = load_benchmark("SOME_BUILT_IN_IDENTIFIER", "/some_directory/")
dataset = benchmark.get_initialized_dataset()
```
Whether loaded dataset is series-based or time-based depends on the benchmark. What can be loaded corresponds to previous datasets.