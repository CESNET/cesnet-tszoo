# Using datasets

This tutorial will look at what you need to use dataset. <br/>
Trying to use dataset you do not have downloaded, will automatically download it.

There currently two supported datasets:

- [CESNET-TimeSeries24][cesnet_timeseries24_page] - supports time-based and series-based
- [CESNET-AGG23][cesnet_agg23_page] - supports only time-based

## Using dataset from benchmark
You can refer to [benchmarks][benchmarks] for more detailed usage.

```python

from cesnet_tszoo.benchmarks import load_benchmark                                                                       

# Imports built-in benchmark
benchmark = load_benchmark(identifier="2e92831cb502", data_root="/some_directory/")
dataset = benchmark.get_initialized_dataset(display_config_details=True, check_errors=False, workers="config")

# Imports custom benchmark
benchmark = load_benchmark(identifier="test2", data_root="/some_directory/")
dataset = benchmark.get_initialized_dataset(display_config_details=True, check_errors=False, workers="config")

```

## Creating dataset
You can refer to [choosing_data][choosing-data] for more detailed data selection via config.

### Using dataset from CESNET_TimeSeries24

```python

from cesnet_tszoo.configs import TimeBasedConfig # For time-based dataset
from cesnet_tszoo.configs import SeriesBasedConfig # For series-based dataset   
from cesnet_tszoo.configs import DisjointTimeBasedConfig # For disjoint-time-based dataset

from cesnet_tszoo.utils.enums import AgreggationType, SourceType # Used for specifying which dataset to use
from cesnet_tszoo.datasets import CESNET_TimeSeries24

# Time-based
time_based_dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.TIME_BASED)
config = TimeBasedConfig(ts_ids=50)
time_based_dataset.set_dataset_config_and_initialize(config)

# Disjoint-time-based
disjoint_dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.DISJOINT_TIME_BASED)
config = DisjointTimeBasedConfig(train_ts=50, val_ts=None, test_ts=None, train_time_period=range(0, 200))
disjoint_dataset.set_dataset_config_and_initialize(config)

# Series-based
series_based_dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/", source_type=SourceType.INSTITUTIONS, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.SERIES_BASED)
config = SeriesBasedDataset(time_period=range(0, 200))
series_based_dataset.set_dataset_config_and_initialize(config)

```

### Using dataset from CESNET_AGG23

```python

from cesnet_tszoo.configs import TimeBasedConfig # For time-based dataset 

from cesnet_tszoo.datasets import CESNET_AGG23

# Using dataset from CESNET_AGG23
# Only time-based
time_based_dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/")
config = TimeBasedConfig(ts_ids=1)
time_based_dataset.set_dataset_config_and_initialize(config)

```