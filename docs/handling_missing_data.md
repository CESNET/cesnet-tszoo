# Handling missing data

This tutorial will look at some configuration options used for handling missing data.

Only time-based will be used, because all methods work almost the same way for other dataset types.

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`handling_missing_data`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/handling_missing_data.ipynb)

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
To see all built-in fillers refer to [`Fillers`](reference_fillers.md#cesnet_tszoo.utils.filler.filler.MeanFiller).

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

To check Filler base class refer to [`Filler`](reference_fillers.md#cesnet_tszoo.utils.filler.filler.Filler)

```python

from cesnet_tszoo.utils.filler import Filler
from cesnet_tszoo.configs import TimeBasedConfig

class CustomFiller(Filler):
    def __init__(self):
        self.last_values = {}
        self.initialized = False

    def __init_attributes(self, batch_values: np.ndarray):
        for name in batch_values.dtype.names:
            self.last_values[name] = None

        self.initialized = True

    def __fill_section(self, values: np.ndarray, missing_mask: np.ndarray, last_values: np.ndarray, name: str) -> np.ndarray:
        if last_values is not None and np.any(missing_mask[0]):
            values[0, missing_mask[0]] = last_values[missing_mask[0]]

        orig_shape = values.shape
        t = orig_shape[0]
        flat_size = int(np.prod(orig_shape[1:]))

        values_2d = values.reshape(t, flat_size)
        mask_2d = missing_mask.reshape(t, flat_size)

        mask = mask_2d.T
        values_t = values_2d.T

        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)

        values_t[mask] = values_t[np.nonzero(mask)[0], idx[mask]]
        values_t = values_t.T

        values = values_2d.reshape(orig_shape)

        self.last_values[name] = np.copy(values[-1])

        return values

    def fill(self, batch_values: np.ndarray, missing_masks: dict[str, np.ndarray], **kwargs) -> np.ndarray:

        if not self.initialized:
            self.__init_attributes(batch_values)

        for name in batch_values.dtype.names:

            values = batch_values[name].view()
            missing_mask = missing_masks[name]
            last_values = self.last_values[name]

            self.__fill_section(values, missing_mask, last_values, name)

        return batch_values

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

## Changing when are missing values handled
- You can change when are `default_values` and filler applied with `preprocess_order` parameter
- `default_values` are always applied before filler and filler considers values filled with `default_values`, as still missing

```python

from cesnet_tszoo.utils.utils.enums import FillerType
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=[1200], train_time_period=range(0, 30), test_time_period=range(30, 80), features_to_take=['n_flows', 'n_packets'],
                         default_values=None, fill_missing_with=FillerType.FORWARD_FILLER, preprocess_order=["handling_anomalies", "filling_gaps", "transforming"])
                                                                        
# Call on time-based dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:


```python

time_based_dataset.update_dataset_config_and_initialize(preprocess_order=["filling_gaps", "handling_anomalies", "transforming"], workers=0)
# Or
time_based_dataset.set_preprocess_order(preprocess_order=["filling_gaps", "handling_anomalies", "transforming"], workers=0)

```