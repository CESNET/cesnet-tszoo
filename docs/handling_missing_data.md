# Handling missing data

This tutorial will look at some configuration options used for handling missing data.

Only time-based will be used, because all methods work almost the same way for series-based.

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`handling_missing_data`](https://github.com/CESNET/cesnet-tszoo/blob/tutorial_notebooks/handling_missing_data.ipynb)

Relevant configuration values:

- `default_values` - Default values for missing data, applied before fillers.
- `fill_missing_with` - Defines how to fill missing values in the dataset. Can pass FillerType enum or custom Filler type.

## Default values
- Default values are set to missing values before filler is used.
- You can change used default values later with `update_dataset_config_and_initialize` or `set_default_values`.

```python

from cesnet_tszoo.configs import TimeBasedConfig

# Default values are provided from used dataset.
config = TimeBasedConfig(ts_ids=[1200], train_time_period=range(0, 30), test_time_period=range(30, 80), features_to_take=['n_flows', 'n_packets'],
                         default_values="default")

# All missing values will be set as None
config = TimeBasedConfig(ts_ids=[1200], train_time_period=range(0, 30), test_time_period=range(30, 80), features_to_take=['n_flows', 'n_packets'],
                         default_values=None)     

# All missing values will be set with 0
config = TimeBasedConfig(ts_ids=[1200], train_time_period=range(0, 30), test_time_period=range(30, 80), features_to_take=['n_flows', 'n_packets'],
                         default_values=0) 

# Using list to specify default values for each used feature
# Position of values in list correspond to order of features in `features_to_take`.
config = TimeBasedConfig(ts_ids=[1200], train_time_period=range(0, 30), test_time_period=range(30, 80), features_to_take=['n_flows', 'n_packets'],
                         default_values=[1, None])       

# Using dictionary with key as name for used feature and value as a default value for missing data
# Dictionary must contain key and value for every feature in `features_to_take`.
config = TimeBasedConfig(ts_ids=[1200], train_time_period=range(0, 30), test_time_period=range(30, 80), features_to_take=['n_flows', 'n_packets'],
                         default_values={"n_flows" : 1, "n_packets": None})                                                                                       

# Call on time-based dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

time_based_dataset.update_dataset_config_and_initialize(default_values="default", workers=0)
# Or
time_based_dataset.set_default_values(default_values="default", workers=0)

```

## Fillers
- Fillers are implemented as classes.
    - You can create your own or use built-in one.
- One filler per time series is created.
- Filler is applied after default values and usually overrides them.
- Fillers in time-based dataset can carry over values from train -> val -> test. Example is in Jupyter notebook.
- You can change used filler later with `update_dataset_config_and_initialize` or `apply_filler`.

### Built-in
To see all built-in fillers refer to [`Fillers`][cesnet_tszoo.utils.filler.MeanFiller].

```python

from cesnet_tszoo.utils.enums import FillerType
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=[1200], train_time_period=range(0, 30), test_time_period=range(30, 80), features_to_take=['n_flows', 'n_packets'],
                         default_values=None, fill_missing_with=FillerType.FORWARD_FILLER)                                                                                

# Call on time-based dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(fill_missing_with=FillerType.FORWARD_FILLER, workers=0)
# Or
time_based_dataset.apply_filler(fill_missing_with=FillerType.FORWARD_FILLER, workers=0)

```

### Custom
You can create your own custom filler, which must derive from 'Filler' base class. 

To check Filler base class refer to [`Filler`][cesnet_tszoo.utils.filler.Filler]

```python

from cesnet_tszoo.utils.filler import Filler
from cesnet_tszoo.configs import TimeBasedConfig

class CustomFiller(Filler):
    def fill(self, batch_values: np.ndarray, existing_indices: np.ndarray, missing_indices: np.ndarray, **kwargs):
        batch_values[missing_indices] = -1

config = TimeBasedConfig(ts_ids=[1200], train_time_period=range(0, 30), test_time_period=range(30, 80), features_to_take=['n_flows', 'n_packets'],
                         default_values=None, fill_missing_with=CustomFiller)                                                                            

# Call on time-based dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:


```python

time_based_dataset.update_dataset_config_and_initialize(fill_missing_with=CustomFiller, workers=0)
# Or
time_based_dataset.apply_filler(CustomFiller, workers=0)

```