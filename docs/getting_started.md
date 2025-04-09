# Getting started

!!! info "Note"
    For a demonstration of usage for simple forecasting refer to Jupyter notebook [`simple_forecasting`](https://github.com/CESNET/cesnet-tszoo/blob/tutorial_notebooks/simple_forecasting.ipynb)

## Code snippets

### Download a dataset
```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType

dataset = CESNET_TimeSeries24.get_dataset("/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, is_series_based=False)
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

#### Using [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset] dataset
```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType
from cesnet_tszoo.configs import TimeBasedConfig

dataset = CESNET_TimeSeries24.get_dataset("/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, is_series_based=False)
config = TimeBasedConfig(
    ts_ids=50, # number of randomly selected time series from dataset
    train_time_period=range(0, 100), 
    val_time_period=range(100, 150), 
    test_time_period=range(150, 250), 
    features_to_take=["n_flows", "n_packets"])
dataset.set_dataset_config_and_initialize(config)
```
Time-based datasets are configured with [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig].
Can load data using:

- [`get_train_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_dataloader], [`get_val_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader], [`get_test_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader], [`get_test_other_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_dataloader], [`get_all_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_all_dataloader]

- [`get_train_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df], [`get_val_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df], [`get_test_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df], [`get_test_other_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_df], [`get_all_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_all_df]

- [`get_train_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy], [`get_val_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy], [`get_test_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy], [`get_test_other_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_numpy], [`get_all_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_all_numpy]

#### Using [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] dataset
```python
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType
from cesnet_tszoo.configs import SeriesBasedConfig

dataset = CESNET_TimeSeries24.get_dataset("/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, is_series_based=True)
config = SeriesBasedConfig(
    time_period=range(0, 250), 
    train_ts=100, # number of randomly selected time series from dataset
    val_ts=30, # number of randomly selected time series from dataset
    test_ts=20, # number of randomly selected time series from dataset
    features_to_take=["n_flows", "n_packets"])
dataset.set_dataset_config_and_initialize(config)
```
Series-based datasets are configured with [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig].
Can load data using:

- [`get_train_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_dataloader], [`get_val_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_dataloader], [`get_test_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_dataloader], [`get_all_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_all_dataloader]

- [`get_train_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_df], [`get_val_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_df], [`get_test_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_df], [`get_all_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_all_df]

- [`get_train_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_numpy], [`get_val_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_numpy], [`get_test_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_numpy], [`get_all_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_all_numpy]

#### Using [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark]
```python
from cesnet_tszoo.benchmarks import load_benchmark

benchmark = load_benchmark("SOME_BUILT_IN_IDENTIFIER", "/some_directory/")
dataset = benchmark.get_initialized_dataset()
```
Whether loaded dataset is series-based or time-based depends on the benchmark. What can be loaded corresponds to previous datasets.