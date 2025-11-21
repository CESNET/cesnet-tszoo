# Loading data

This tutorial will look at some configuration options used for loading data.

Each dataset type will have its own part because of multiple differences of available configuration values.

## [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`time_based_loading_data`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/time_based_loading_data.ipynb)

Relevant configuration values:

- `sliding_window_size` - Number of times in one window.
- `sliding_window_prediction_size` - Number of times to predict from `sliding_window_size`.
- `sliding_window_step` - Number of times to move by after each window.
- `set_shared_size` - How much times should time periods share. Order of sharing is training set < validation set < test set.
- `train_batch_size`/`val_batch_size`/`test_batch_size`/`all_batch_size` - How many times for every time series will be in one batch (differs when sliding window is used).
- `train_workers`/`val_workers`/`test_workers`/`all_workers` - Defines how many workers (processes) will be used for loading specific set.

### Loading data with DataLoader
- Load data using Pytorch Dataloader.
- Affected by workers and batch sizes.
- Last batch is never dropped (unless sliding window is used)
- Returned batch shape changes when used `time_format` is TimeFormat.DATETIME compare to other time formats. Check Jupyter notebook for details.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=54, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take="all", time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0,
                         train_batch_size=32, val_batch_size=64, test_batch_size=128, all_batch_size=128)

# Call on time-based dataset to use created config -> must be called before attempting to load data.
time_based_dataset.set_dataset_config_and_initialize(config)

# train_time_period must be set
dataloader = time_based_dataset.get_train_dataloader(workers="config")

# val_time_period must be set
dataloader = time_based_dataset.get_val_dataloader(workers="config")

# test_time_period must be set
dataloader = time_based_dataset.get_test_dataloader(workers="config")

# Always usable
dataloader = time_based_dataset.get_all_dataloader(workers="config")

# Example of usage -> loads all batches into list
batches = []

for batch in tqdm(dataloader):
    # batch is a Numpy array of shape (ts_ids, batch_size, features_to_take + used ids)
    batches.append(batch)

```

You can also change set batch sizes later with `update_dataset_config_and_initialize` or `set_batch_sizes`.

```python

time_based_dataset.update_dataset_config_and_initialize(train_batch_size=33, val_batch_size=65, test_batch_size="config", all_batch_size="config")
# Or
time_based_dataset.set_batch_sizes(train_batch_size=33, val_batch_size=65, test_batch_size="config", all_batch_size="config")

```

You can also change set workers later with `update_dataset_config_and_initialize` or `set_workers`.

```python

time_based_dataset.update_dataset_config_and_initialize(train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0)
# Or
time_based_dataset.set_workers(train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0)

```

You can also specify which time series to load from set.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=54, train_time_period=[0,1,2,3,4], features_to_take="all", time_format=TimeFormat.ID_TIME,
                         train_workers=0, train_batch_size=32,)

time_based_dataset.set_dataset_config_and_initialize(config)

# train_time_period must be set; load time series with id == 1
dataloader = time_based_dataset.get_train_dataloader(ts_id=1, workers="config")

# Example of usage -> loads all batches into list
batches = []

for batch in tqdm(dataloader):
    # batch is a Numpy array of shape (1, batch_size, features_to_take + used ids)
    batches.append(batch)

```

#### Sliding window
- When `sliding_window_prediction_size` is set then `sliding_window_size` must be set too if you want to use sliding window.
- Batch sizes are used for background caching.
- You can modify sliding window step size with `sliding_window_step`
- You can use `set_shared_size` to set how many times time periods should share.
    - `val_time_period` takes from `train_time_period`
    - `test_time_period` takes from `val_time_period` or `train_time_period`

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=54, train_time_period=range(0, 1000), val_time_period=range(1000, 1500), test_time_period=range(1500, 2000), features_to_take=["n_flows"], time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0,
                         train_batch_size=32, val_batch_size=64, test_batch_size=128, all_batch_size=128,
                         sliding_window_size=22, sliding_window_prediction_size=2, sliding_window_step=2, set_shared_size=0.05)

time_based_dataset.set_dataset_config_and_initialize(config)

dataloader = time_based_dataset.get_train_dataloader(workers="config")

# Example of usage -> loads all batches into list
batches = []

for sliding_window, sliding_window_prediction in tqdm(dataloader):
    # sliding_window is a Numpy array of shape (ts_ids, sliding_window_size, features_to_take + used ids)
    # sliding_window_prediction is a Numpy array of shape (ts_ids, sliding_window_prediction_size, features_to_take + used ids)
    batches.append((sliding_window, sliding_window_prediction))    

```

You can also change sliding window parameters later with `update_dataset_config_and_initialize` or `set_sliding_window`.

```python

time_based_dataset.update_dataset_config_and_initialize(sliding_window_size=22, sliding_window_prediction_size=3, sliding_window_step="config", set_shared_size="config", workers=0)
# Or
time_based_dataset.set_sliding_window(sliding_window_size=22, sliding_window_prediction_size=3, sliding_window_step="config", set_shared_size="config", workers=0)

```

### Loading data as Dataframe
- Batch size has no effect.
- Sliding window has no effect.
- Returns every time series in `ts_ids` with sets specified time period.
- Data is returned as Pandas Dataframe.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=54, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take="all", time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0)

time_based_dataset.set_dataset_config_and_initialize(config)

# train_time_period must be set
df = time_based_dataset.get_train_df(as_single_dataframe=True, workers="config") # loads time series from ts_ids of train_time_period into one Pandas Dataframe
dfs = time_based_dataset.get_train_df(as_single_dataframe=False, workers="config") # loads time series from ts_ids of train_time_period into seperate Pandas Dataframes

# val_time_period must be set
df = time_based_dataset.get_val_df(as_single_dataframe=True, workers="config") # loads time series from ts_ids of val_time_period into one Pandas Dataframe
dfs = time_based_dataset.get_val_df(as_single_dataframe=False, workers="config") # loads time series from ts_ids of val_time_period into seperate Pandas Dataframes

# test_time_period must be set
df = time_based_dataset.get_test_df(as_single_dataframe=True, workers="config") # loads time series from ts_ids of test_time_period into one Pandas Dataframe
dfs = time_based_dataset.get_test_df(as_single_dataframe=False, workers="config") # loads time series from ts_ids of test_time_period into seperate Pandas Dataframes

# Always usable
df = time_based_dataset.get_all_df(as_single_dataframe=True, workers="config") # loads time series from ts_ids of all time period into one Pandas Dataframe
dfs = time_based_dataset.get_all_df(as_single_dataframe=False, workers="config") # loads time series from ts_ids of all time period into seperate Pandas Dataframes

```

### Loading data as singular Numpy array 
- Batch size has no effect.
- Sliding window has no effect.
- Returns every time series in `ts_ids` with sets specified time period.
- Data is returned as one Numpy array.
- Follows similar rules to Dataloader batches, regarding shape (excluding sliding window parameters).

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=54, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take="all", time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0)

time_based_dataset.set_dataset_config_and_initialize(config)

# train_time_period must be set
numpy_array = time_based_dataset.get_train_numpy(workers="config")

# val_time_period must be set
numpy_array = time_based_dataset.get_val_numpy(workers="config")

# test_time_period must be set
numpy_array = time_based_dataset.get_test_numpy(workers="config")

# Always usable
numpy_array = time_based_dataset.get_all_numpy(workers="config")

```

## [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`disjoint_time_based_loading_data`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/disjoint_time_based_loading_data.ipynb)

Relevant configuration values:

- `sliding_window_size` - Number of times in one window.
- `sliding_window_prediction_size` - Number of times to predict from `sliding_window_size`.
- `sliding_window_step` - Number of times to move by after each window.
- `set_shared_size` - How much times should time periods share. Order of sharing is training set < validation set < test set.
- `train_batch_size`/`val_batch_size`/`test_batch_size` - How many times for every time series will be in one batch (differs when sliding window is used).
- `train_workers`/`val_workers`/`test_workers` - Defines how many workers (processes) will be used for loading specific set.

### Loading data with DataLoader
- Load data using Pytorch Dataloader.
- Affected by workers and batch sizes.
- Last batch is never dropped (unless sliding window is used)
- Returned batch shape changes when used `time_format` is TimeFormat.DATETIME compare to other time formats. Check Jupyter notebook for details.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import DisjointTimeBasedConfig

config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take="all", time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, init_workers=0,
                         train_batch_size=32, val_batch_size=64, test_batch_size=128)

# Call on disjoint-time-based dataset to use created config -> must be called before attempting to load data.
disjoint_dataset.set_dataset_config_and_initialize(config)

# train_ts and train_time_period must be set
dataloader = disjoint_dataset.get_train_dataloader(workers="config")

# val_ts and val_time_period must be set
dataloader = disjoint_dataset.get_val_dataloader(workers="config")

# test_ts and test_time_period must be set
dataloader = disjoint_dataset.get_test_dataloader(workers="config")

# Example of usage -> loads all batches into list
batches = []

for batch in tqdm(dataloader):
    # batch is a Numpy array of shape (train_ts/val_ts/test_ts, batch_size, features_to_take + used ids)
    batches.append(batch)

```

You can also change set batch sizes later with `update_dataset_config_and_initialize` or `set_batch_sizes`.

```python

disjoint_dataset.update_dataset_config_and_initialize(train_batch_size=33, val_batch_size=65, test_batch_size="config")
# Or
disjoint_dataset.set_batch_sizes(train_batch_size=33, val_batch_size=65, test_batch_size="config")

```

You can also change set workers later with `update_dataset_config_and_initialize` or `set_workers`.

```python

disjoint_dataset.update_dataset_config_and_initialize(train_workers=0, val_workers=0, test_workers=0, init_workers=0)
# Or
disjoint_dataset.set_workers(train_workers=0, val_workers=0, test_workers=0, init_workers=0)

```

You can also specify which time series to load from set.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import DisjointTimeBasedConfig

config = DisjointTimeBasedConfig(train_ts=[177, 176, 319, 267], val_ts=None, test_ts=None, train_time_period=0.5, features_to_take="all", time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, init_workers=0,
                         train_batch_size=32, val_batch_size=64, test_batch_size=128)

disjoint_dataset.set_dataset_config_and_initialize(config)

# train_time_period must be set; load time series with id == 177
dataloader = disjoint_dataset.get_train_dataloader(ts_id=177, workers="config")

# Example of usage -> loads all batches into list
batches = []

for batch in tqdm(dataloader):
    # batch is a Numpy array of shape (1, batch_size, features_to_take + used ids)
    batches.append(batch)

```

#### Sliding window
- When `sliding_window_prediction_size` is set then `sliding_window_size` must be set too if you want to use sliding window.
- Batch sizes are used for background caching.
- You can modify sliding window step size with `sliding_window_step`
- You can use `set_shared_size` to set how many times time periods should share.
    - `val_time_period` takes from `train_time_period`
    - `test_time_period` takes from `val_time_period` or `train_time_period`

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import DisjointTimeBasedConfig

config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=range(0, 1000), val_time_period=range(1000, 1500), test_time_period=range(1500, 2000), 
                         features_to_take=["n_flows"], time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, init_workers=0,
                         train_batch_size=32, val_batch_size=64, test_batch_size=128,
                         sliding_window_size=22, sliding_window_prediction_size=2, sliding_window_step=2, set_shared_size=0.05)

disjoint_dataset.set_dataset_config_and_initialize(config)

dataloader = disjoint_dataset.get_train_dataloader(workers="config")

# Example of usage -> loads all batches into list
batches = []

for sliding_window, sliding_window_prediction in tqdm(dataloader):
    # sliding_window is a Numpy array of shape (train_ts/val_ts/test_ts, sliding_window_size, features_to_take + used ids)
    # sliding_window_prediction is a Numpy array of shape (train_ts/val_ts/test_ts, sliding_window_prediction_size, features_to_take + used ids)
    batches.append((sliding_window, sliding_window_prediction))    

```

You can also change sliding window parameters later with `update_dataset_config_and_initialize` or `set_sliding_window`.

```python

disjoint_dataset.update_dataset_config_and_initialize(sliding_window_size=22, sliding_window_prediction_size=3, sliding_window_step="config", set_shared_size="config", workers=0)
# Or
disjoint_dataset.set_sliding_window(sliding_window_size=22, sliding_window_prediction_size=3, sliding_window_step="config", set_shared_size="config", workers=0)

```

### Loading data as Dataframe
- Batch size has no effect.
- Sliding window has no effect.
- Returns every time series in `train_ts`/`val_ts`/`test_ts` with sets specified time period.
- Data is returned as Pandas Dataframe.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import DisjointTimeBasedConfig

config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take="all", time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, init_workers=0)

disjoint_dataset.set_dataset_config_and_initialize(config)

# train_ts and train_time_period must be set
df = disjoint_dataset.get_train_df(as_single_dataframe=True, workers="config") # loads time series from train_ts of train_time_period into one Pandas Dataframe
dfs = disjoint_dataset.get_train_df(as_single_dataframe=False, workers="config") # loads time series from train_ts of train_time_period into seperate Pandas Dataframes

# val_ts and val_time_period must be set
df = disjoint_dataset.get_val_df(as_single_dataframe=True, workers="config") # loads time series from val_ts of val_time_period into one Pandas Dataframe
dfs = disjoint_dataset.get_val_df(as_single_dataframe=False, workers="config") # loads time series from val_ts of val_time_period into seperate Pandas Dataframes

# test_ts and test_time_period must be set
df = disjoint_dataset.get_test_df(as_single_dataframe=True, workers="config") # loads time series from test_ts of test_time_period into one Pandas Dataframe
dfs = disjoint_dataset.get_test_df(as_single_dataframe=False, workers="config") # loads time series from test_ts of test_time_period into seperate Pandas Dataframes

```

### Loading data as singular Numpy array 
- Batch size has no effect.
- Sliding window has no effect.
- Returns every time series in `train_ts`/`val_ts`/`test_ts` with sets specified time period.
- Data is returned as one Numpy array.
- Follows similar rules to Dataloader batches, regarding shape (excluding sliding window parameters).

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import DisjointTimeBasedConfig

config = DisjointTimeBasedConfig(train_ts=0.5, val_ts=0.2, test_ts=0.1, train_time_period=0.5, val_time_period=0.3, test_time_period=0.2, features_to_take="all", time_format=TimeFormat.ID_TIME,
                         train_workers=0, val_workers=0, test_workers=0, init_workers=0)

disjoint_dataset.set_dataset_config_and_initialize(config)

# train_ts and train_time_period must be set
numpy_array = disjoint_dataset.get_train_numpy(workers="config")

# val_ts and val_time_period must be set
numpy_array = disjoint_dataset.get_val_numpy(workers="config")

# test_ts and test_time_period must be set
numpy_array = disjoint_dataset.get_test_numpy(workers="config")


```

## [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`series_based_loading_data`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/series_based_loading_data.ipynb)

Relevant configuration values:

- `train_batch_size`/`val_batch_size`/`test_batch_size`/`all_batch_size` - How many time series will be in one batch.
- `train_workers`/`val_workers`/`test_workers`/`all_workers` - Defines how many workers (processes) will be used for loading specific set.
- `train_dataloader_order` - Affects order of time series in loaded batch.
- `random_state` - When set, batches will be same when using random order type for `train_dataloader_order`. 

### Loading data with DataLoader
- Load data using Pytorch Dataloader.
- Affected by workers and batch sizes.
- Last batch is never dropped.
- Returned batch shape changes when used `time_format` is TimeFormat.DATETIME compare to other time formats. Check Jupyter notebook for details.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import SeriesBasedConfig

config = SeriesBasedConfig(time_period=0.5, train_ts=54, val_ts=25, test_ts=10, features_to_take=["n_flows"], time_format=TimeFormat.ID_TIME,
                           train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0,
                           train_batch_size=32, val_batch_size=64, test_batch_size=128, all_batch_size=128)

# Call on series-based dataset to use created config -> must be called before attempting to load data.
series_based_dataset.set_dataset_config_and_initialize(config)

# train_ts must be set
dataloader = series_based_dataset.get_train_dataloader(workers="config")

# val_ts must be set
dataloader = series_based_dataset.get_val_dataloader(workers="config")

# test_ts must be set
dataloader = series_based_dataset.get_test_dataloader(workers="config")

# Always usable
dataloader = series_based_dataset.get_all_dataloader(workers="config")

# Example of usage -> loads all batches into list
batches = []

for batch in tqdm(dataloader):
    # batch is a Numpy array of shape (batch_size, time_period, features_to_take + used ids)
    batches.append(batch)

```

You can also change set batch sizes later with `update_dataset_config_and_initialize` or `set_batch_sizes`.

```python

series_based_dataset.update_dataset_config_and_initialize(train_batch_size=33, val_batch_size=65, test_batch_size="config", all_batch_size="config")
# Or
series_based_dataset.set_batch_sizes(train_batch_size=33, val_batch_size=65, test_batch_size="config", all_batch_size="config")

```

You can also change set workers later with `update_dataset_config_and_initialize` or `set_workers`.

```python

series_based_dataset.update_dataset_config_and_initialize(train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0)
# Or
series_based_dataset.set_workers(train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0)

```

You can also specify which time series to load from set.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import SeriesBasedConfig

config = SeriesBasedConfig(time_period=0.5, train_ts=[177, 176, 319, 267], features_to_take=["n_flows"], time_format=TimeFormat.ID_TIME,
                           train_workers=0, train_batch_size=32)

series_based_dataset.set_dataset_config_and_initialize(config)

# train_ts must be set; load time series with id == 176
dataloader = series_based_dataset.get_train_dataloader(ts_id=176, workers="config")

# Example of usage -> loads one whole time series
batches = []

for batch in tqdm(dataloader):
    # batch is a Numpy array of shape (1, time_period, features_to_take + used ids)
    batches.append(batch)

```

### Loading data as Dataframe
- Batch size has no effect.
- Returns every time series in set with specified `time_period`.
- Data is returned as Pandas Dataframe.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import SeriesBasedConfig

config = SeriesBasedConfig(time_period=0.5, train_ts=54, val_ts=25, test_ts=10, features_to_take="all", time_format=TimeFormat.ID_TIME,
                           train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0)

series_based_dataset.set_dataset_config_and_initialize(config)

# train_ts must be set
df = series_based_dataset.get_train_df(as_single_dataframe=True, workers="config") # loads every time series from train_ts with time_period into one Pandas Dataframe
dfs = series_based_dataset.get_train_df(as_single_dataframe=False, workers="config") # loads every time series from train_ts with time_period into seperate Pandas Dataframe

# val_ts must be set
df = series_based_dataset.get_val_df(as_single_dataframe=True, workers="config") # loads every time series with from val_ts time_period into one Pandas Dataframe
dfs = series_based_dataset.get_val_df(as_single_dataframe=False, workers="config") # loads every time series from val_ts with time_period into seperate Pandas Dataframe

# test_ts must be set
df = series_based_dataset.get_test_df(as_single_dataframe=True, workers="config") # loads every time series from test_ts with time_period into one Pandas Dataframe
dfs = series_based_dataset.get_test_df(as_single_dataframe=False, workers="config") # loads every time series from test_ts with time_period into seperate Pandas Dataframe

# Always usable
df = series_based_dataset.get_all_df(as_single_dataframe=True, workers="config") # loads every time series from all set with time_period into one Pandas Dataframe
dfs = series_based_dataset.get_all_df(as_single_dataframe=False, workers="config") # loads every time series from all set with time_period into seperate Pandas Dataframe

```

### Loading data as singular Numpy array 
- Batch size has no effect.
- Returns every time series in set with specified `time_period`.
- Data is returned as one Numpy array.
- Follows similar rules to Dataloader batches, regarding shape.

```python

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.configs import SeriesBasedConfig

config = SeriesBasedConfig(time_period=0.5, train_ts=54, val_ts=25, test_ts=10, features_to_take="all", time_format=TimeFormat.ID_TIME,
                           train_workers=0, val_workers=0, test_workers=0, all_workers=0, init_workers=0)

series_based_dataset.set_dataset_config_and_initialize(config)

# train_ts must be set
numpy_array = series_based_dataset.get_train_numpy(workers="config")

# val_ts must be set
numpy_array = series_based_dataset.get_val_numpy(workers="config")

# test_ts must be set
numpy_array = series_based_dataset.get_test_numpy(workers="config")

# Always usable
numpy_array = series_based_dataset.get_all_numpy(workers="config")

``` 