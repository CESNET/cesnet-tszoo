# Choosing data

This tutorial will look at some configuration options for choosing data you wish to load. 

Each dataset type will have its own part because of multiple differences of available configuration values.

## [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset) dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`time_based_choosing_data`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/time_based_choosing_data.ipynb)

Relevant configuration values:

- `ts_ids` - Defines which time series IDs are used for train/val/test/all.
- `train_time_period`/`val_time_period`/`test_time_period` - Defines time periods for train/val/test sets.
- `features_to_take` - Defines which features are used.
- `include_time` - If True, time data is included in the returned values.
- `include_ts_id` - If True, time series IDs are included in the returned values.
- `time_format` - Format for the returned time data.
- `random_state` - Fixes randomness for reproducibility when setting `ts_ids`

### Selecting which time series to load
- Sets time series that will be used for train/val/test/all sets

```python

from cesnet_tszoo.configs import TimeBasedConfig

# Sets time series used in sets with count. Chosen randomly from available time series.
# Affected by random_state.
config = TimeBasedConfig(ts_ids=54, random_state = 111)

# Sets time series used in sets with percentage of time series in dataset. Chosen randomly from available time series.
# Affected by random_state.
config = TimeBasedConfig(ts_ids=0.1, random_state = 111)

# Sets ts_ids with specific time series
config = TimeBasedConfig(ts_ids=[0,1,2,3,4,5])

# Call on time-based dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

```

### Creating train/val/test sets
- Sets time period in set for every time series in `ts_ids`
- You can leave any set value set as None.
- Can use `nan_threshold` to set how many nan values will be tolerated.
    - `nan_threshold` = 1.0, means that time series can be completely empty.
    - is applied after sets.
    - Is checked seperately for every set.
- Sets must follow these rules:
    - Used time periods must be connected.
    - Sets can share subset of times.
    - start of `train_time_period` < start of `val_time_period` < start of `test_time_period`.

```python

from datetime import datetime

from cesnet_tszoo.configs import TimeBasedConfig

# Sets sets as range of time indices.
config = TimeBasedConfig(ts_ids=54, train_time_period=range(0, 2000), val_time_period=range(2000, 4000), test_time_period=range(4000, 5000))

# Sets sets with tuple of datetime objects.
# Datetime objects are expected to be of UTC.
config = TimeBasedConfig(ts_ids=54, train_time_period=(datetime(2023, 10, 9, 0), datetime(2023, 11, 9, 23)), val_time_period=(datetime(2023, 11, 9, 23), datetime(2023, 12, 9, 23)), test_time_period=(datetime(2023, 12, 9, 23), datetime(2023, 12, 25, 23)))

# Sets sets a percentage of whole time period from dataset.
# Always starts from first time.
config = TimeBasedConfig(ts_ids=54, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2)

time_based_dataset.set_dataset_config_and_initialize(config)

```

### Selecting features
- Affects which features will be returned when loading data.
- Setting `include_time` as True will add time to features that return when loading data.
- Setting `include_ts_id` as True will add time series id to features that return when loading data.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=54, features_to_take="all")

config = TimeBasedConfig(ts_ids=54, features_to_take=["n_flows", "n_packets"])

config = TimeBasedConfig(ts_ids=54, features_to_take=["n_flows", "n_packets"], include_time=True, include_ts_id=True, time_format=TimeFormat.ID_TIME)

time_based_dataset.set_dataset_config_and_initialize(config)

```

### Selecting all set

- Contains time series from `ts_ids`.

```python

from cesnet_tszoo.configs import TimeBasedConfig

# All set will contain whole time period of dataset.
config = TimeBasedConfig(ts_ids=54, train_time_period=None, val_time_period=None, test_time_period=None)

# All set will contain total time period of train + val + test.
config = TimeBasedConfig(ts_ids=54, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2)

time_based_dataset.set_dataset_config_and_initialize(config)

```

## [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset) dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`disjoint_time_based_choosing_data`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/disjoint_time_based_choosing_data.ipynb)

Relevant configuration values:

- `train_ts`/`val_ts`/`test_ts` - Defines time series for train/val/test.
- `train_time_period`/`val_time_period`/`test_time_period` - Defines time periods for train/val/test sets.
- `features_to_take` - Defines which features are used.
- `include_time` - If True, time data is included in the returned values.
- `include_ts_id` - If True, time series IDs are included in the returned values.
- `time_format` - Format for the returned time data.
- `random_state` - Fixes randomness for reproducibility when setting `train_ts`/`val_ts`/`test_ts`

### Selecting which time series to load
- Sets time series that will be used for train/val/test/all sets

```python

from cesnet_tszoo.configs import DisjointTimeBasedConfig

# Sets time series used in sets with count. Chosen randomly from available time series.
# Affected by random_state.
config = DisjointTimeBasedConfig(train_ts=100, val_ts=50, test_ts=20, train_time_period=0.7, val_time_period=0.2, test_time_period=0.1, random_state = 111)

# Sets time series used in sets with percentage of time series in dataset. Chosen randomly from available time series.
# Affected by random_state.
config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=0.7, val_time_period=0.2, test_time_period=0.1, random_state = 111)

# Sets with specific time series
config = DisjointTimeBasedConfig(train_ts=[0], val_ts=[1], test_ts=[2], train_time_period=0.7, val_time_period=0.2, test_time_period=0.1, random_state = 111)

# Call on disjoint-time-based dataset to use created config
disjoint_dataset.set_dataset_config_and_initialize(config)

```

### Selecting which time period to use for each set
- Sets time period for every set and their time series
- `train_time_period` is used for `train_ts`
- `val_time_period` is used for `val_ts`
- `test_time_period` is used for `test_ts`
- Either both time series and their time period must be set or both has to be None
- Can use `nan_threshold` to set how many nan values will be tolerated for time series and their time period.
    - `nan_threshold` = 1.0, means that time series can be completely empty.
    - is applied after sets.
    - Is checked seperately for every set.
- Sets must follow these rules:
    - Used time periods must be connected.
    - Sets can share subset of times.
    - start of `train_time_period` < start of `val_time_period` < start of `test_time_period`.

```python

from datetime import datetime

from cesnet_tszoo.configs import DisjointTimeBasedConfig

# Sets sets as range of time indices.
config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=range(0, 2000), val_time_period=range(2000, 4000), test_time_period=range(4000, 5000))

# Sets sets with tuple of datetime objects.
# Datetime objects are expected to be of UTC.
config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=(datetime(2023, 10, 9, 0), datetime(2023, 11, 9, 23)), val_time_period=(datetime(2023, 11, 9, 23), datetime(2023, 12, 9, 23)), test_time_period=(datetime(2023, 12, 9, 23), datetime(2023, 12, 25, 23)))

# Sets sets a percentage of whole time period from dataset.
# Always starts from first time.
config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2)

disjoint_dataset.set_dataset_config_and_initialize(config)

```

### Selecting features
- Affects which features will be returned when loading data.
- Setting `include_time` as True will add time to features that return when loading data.
- Setting `include_ts_id` as True will add time series id to features that return when loading data.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import DisjointTimeBasedConfig

config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take="all")

config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take=["n_flows", "n_packets"])

config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take=["n_flows", "n_packets"], include_time=True, include_ts_id=True, time_format=TimeFormat.ID_TIME)

disjoint_dataset.set_dataset_config_and_initialize(config)

```

## [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`series_based_choosing_data`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/series_based_choosing_data.ipynb)

Relevant configuration values:

- `time_period` - Defines the time period for train/val/test/all sets.
- `train_ts`/`val_ts`/`test_ts` - Defines time series for train/val/test
- `features_to_take` - Defines which features are used.
- `include_time` - If True, time data is included in the returned values.
- `include_ts_id` - If True, time series IDs are included in the returned values.
- `time_format` - Format for the returned time data.
- `random_state` - Fixes randomness for reproducibility when setting `train_ts`, `val_ts`, `test_ts`.

### Selecting time period
- `time_period` sets time period for all sets (used time series).

```python

from datetime import datetime

from cesnet_tszoo.configs import SeriesBasedConfig

# Sets time period for time series as a whole time period from dataset.
config = SeriesBasedConfig(time_period="all")

# Sets time period for time series as range of time indices.
config = SeriesBasedConfig(time_period=range(0, 2000))

# Sets time period for time series with tuple of datetime objects.
# Datetime objects are expected to be of UTC.
config = SeriesBasedConfig(time_period=(datetime(2023, 10, 9, 0), datetime(2023, 11, 9, 23)))

# Sets time period for time series as a percentage of whole time period from dataset.
# Always starts from first time.
config = SeriesBasedConfig(time_period=0.5)

# Call on series-based dataset to use created config
series_based_dataset.set_dataset_config_and_initialize(config)

```

### Creating train/val/test sets
- Sets how many time series will be in each set.
- You can leave any set value set as None.
- Each set must have unique time series
- Can use `nan_threshold` to set how many nan values will be tolerated.
    - `nan_threshold` = 1.0, means that time series can be completely empty.
    - is applied after sets.

```python

from cesnet_tszoo.configs import SeriesBasedConfig

# Sets time series in set with count. Chosen randomly from available time series.
# Each set will contain unique time series.
# Affected by random_state.
config = SeriesBasedConfig(time_period=0.5, train_ts=54, val_ts=25, test_ts=10, random_state=None, nan_threshold=1.0)

# Sets time series in set with percentage of time series in dataset. Chosen randomly from available time series.
# Each set will contain unique time series.
# Affected by random_state.
config = SeriesBasedConfig(time_period=0.5, train_ts=0.5, val_ts=0.2, test_ts=0.1, random_state=None, nan_threshold=1.0)

# Sets sets with specific time series
config = SeriesBasedConfig(time_period=0.5, train_ts=[0,1,2,3,4], val_ts=[5,6,7,8,9], test_ts=[10,11,12,13,14], nan_threshold=1.0)

series_based_dataset.set_dataset_config_and_initialize(config)

```

### Selecting features
- Affects which features will be returned when loading data.
- Setting `include_time` as True will add time to features that return when loading data.
- Setting `include_ts_id` as True will add time series id to features that return when loading data.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import SeriesBasedConfig

config = SeriesBasedConfig(time_period=0.5, features_to_take="all")

config = SeriesBasedConfig(time_period=0.5, features_to_take=["n_flows", "n_packets"])

config = SeriesBasedConfig(time_period=0.5, features_to_take=["n_flows", "n_packets"], include_time=True, include_ts_id=True, time_format=TimeFormat.ID_TIME)

# Call on series-based dataset to use created config
series_based_dataset.set_dataset_config_and_initialize(config)

```

### Selecting all set

```python

from cesnet_tszoo.configs import SeriesBasedConfig

# All set will contain all time series from dataset.
config = SeriesBasedConfig(time_period=0.5, train_ts=None, val_ts=None, test_ts=None)

# All set will contain all time series that were set by other sets.
config = SeriesBasedConfig(time_period=0.5, train_ts=54, val_ts=25, test_ts=10)

series_based_dataset.set_dataset_config_and_initialize(config)

```