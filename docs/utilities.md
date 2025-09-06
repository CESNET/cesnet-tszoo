# Utilities

This tutorial will look at various utilities.

Only time-based will be used, because all methods work almost the same way for other dataset types.

!!! info "Note"
    For every option and more detailed examples refer to Jupyter notebook [`utilities`](https://github.com/CESNET/cesnet-tszoo/blob/main/tutorial_notebooks/utilities.ipynb)

## Setting logger
CESNET TS-Zoo uses logger, but without setting config below, it wont log anything.

```python

import logging                                                                       

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

```

## Checking errors
- Goes through all data in dataset to check whether everything is in correct state,
- Can be called when creating dataset or with method `check_errors` on already create dataset.
- Recommended to call at least once after download

```python
from cesnet_tszoo.utils.enums import AgreggationType, SourceType
from cesnet_tszoo.datasets import CESNET_TimeSeries24                                                                

# Can be called at dataset creation
time_based_dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/", source_type=SourceType.IP_ADDRESSES_SAMPLE, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.TIME_BASED, check_errors=True)

# Or after it
time_based_dataset.check_errors()

```

## Dataset details
### Displaying all data about selected dataset
Displays available times, time series, features with their default values, additional data provided by dataset.

```python

dataset.display_dataset_details()

```

### Get list of available features

```python

dataset.get_feature_names()

```

### Get numpy array of available dataset time series indices

```python

dataset.get_available_ts_indices()

```

### Get dictionary of related set data
Returns all data in dictionary related to set.

```python

from cesnet_tszoo.configs import TimeBasedConfig   

config = TimeBasedConfig(20, train_time_period=0.5)
time_based_dataset.set_dataset_config_and_initialize(config, workers=0, display_config_details=False)

time_based_dataset.get_data_about_set(about=SplitType.TRAIN)

```

## Displaying config details
Can be called when calling `set_dataset_config_and_initialize` or after it with `display_config`

```python

from cesnet_tszoo.configs import TimeBasedConfig   

config = TimeBasedConfig(20)

# Can be called during initialization
time_based_dataset.set_dataset_config_and_initialize(config, workers=0, display_config_details=True)

# Or after it
time_based_dataset.display_config()

```

## Plotting
- Uses [`Plotly`](https://plotly.com/python/) library.
- You can plot specific time series with method `plot`
- You can set `ts_id` to any time series id used in config
- Config must be set before using

```python

# Features will be taken from config
dataset.plot(ts_id=10, plot_type="line", features="config", feature_per_plot=True, time_format="datetime")

# Specifies features as list... features must be set in used config
dataset.plot(ts_id=10, plot_type="line", features=["n_flows", "n_packets"], feature_per_plot=True, time_format="datetime")

# Can specify single feature... still must be set in used config
dataset.plot(ts_id=10, plot_type="line", features="n_flows", feature_per_plot=True, time_format="datetime")

```

## Get additional data
- You can check whether dataset has additional data, with method `display_dataset_details`.

```python
from cesnet_tszoo.utils.enums import AgreggationType, SourceType
from cesnet_tszoo.datasets import CESNET_TimeSeries24                                                                

time_based_dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/", source_type=SourceType.IP_ADDRESSES_SAMPLE, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.TIME_BASED, display_details=True)

# Available additional data in CESNET_TimeSeries24 database
time_based_dataset.get_additional_data('ids_relationship')
time_based_dataset.get_additional_data('weekends_and_holidays')

```

## Get fitted transformers
Returns used transformer/s that are used for transforming data.

```python

dataset.get_transformers()

```