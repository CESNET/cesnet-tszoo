# Using custom handlers

This tutorial will look at some configuration options for using custom handlers.

Only time-based will be used, because all methods work almost the same way for other dataset types.

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`using_custom_handlers`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/using_custom_handlers.ipynb)

Relevant configuration values:

- `preprocess_order` - Mainly used for changing order of preprocesses. Is also used as a way of adding custom handlers by adding type their type between the preprocesses.

## Custom handlers
- Custom handlers are implemented as a class
    - Their main purpose is to allow creation of custom preprocessing steps
- There are three types of custom handlers: `AllSeriesCustomHandler`, `PerSeriesCustomHandler`, `NoFitCustomHandler`
- Custom handlers can be used by adding their type to `preprocessing_order` -> which will also define when they will be applied
- All custom handler types allow specifying to which set they can be applied
- You can change used custom handlers later with `update_dataset_config_and_initialize` or `set_preprocess_order`, by modifying `preprocessing_order` parameter.
- Take care that all custom handlers should be imported from other file when while using this library in Jupyter notebook. When not importing from other file/s use workers == 0.

### AllSeriesCustomHandler
- You can refer to [`AllSeriesCustomHandler`](reference_custom_handlers.md#cesnet_tszoo.utils.custom_handler.custom_handler.AllSeriesCustomHandler)
- One instance is created for all time series
- Must always be fitted on train set before use

```python

import numpy as np
from cesnet_tszoo.utils.custom_handler import AllSeriesCustomHandler
from cesnet_tszoo.configs import TimeBasedConfig

class AllFitTest(AllSeriesCustomHandler):

    def __init__(self):
        self.count = 0
        super().__init__()

    def partial_fit(self, data: np.ndarray) -> None:
        self.count += 1

    def apply(self, data: np.ndarray) -> np.ndarray:
        data[:, :] = self.count
        return data

    @staticmethod
    def get_target_sets():
        return ["train"]
          

config = TimeBasedConfig(ts_ids=500, train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, features_to_take=['n_flows', 'n_packets'],
                        nan_threshold=0.5, random_state=1500, preprocess_order=["handling_anomalies", "filling_gaps", "transforming", AllFitTest])

time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(preprocess_order=["handling_anomalies", AllFitTest, "filling_gaps", "transforming"], workers=0)
# Or
time_based_dataset.set_preprocess_order(preprocess_order=["handling_anomalies", AllFitTest, "filling_gaps", "transforming"], workers=0)

```

### PerSeriesCustomHandler
- You can refer to [`PerSeriesCustomHandler`](reference_custom_handlers.md#cesnet_tszoo.utils.custom_handler.custom_handler.PerSeriesCustomHandler)
- One instance is created per time series
- Must always be fitted on train set before use
- Supported only for Time-Based dataset

```python

import numpy as np
from cesnet_tszoo.utils.custom_handler import PerSeriesCustomHandler
from cesnet_tszoo.configs import TimeBasedConfig

class PerFitTest(PerSeriesCustomHandler):

    def __init__(self):
        self.count = 0
        super().__init__()

    def fit(self, data: np.ndarray) -> None:
        self.count += 1

    def apply(self, data: np.ndarray) -> np.ndarray:
        data[:, :] = self.count
        return data

    @staticmethod
    def get_target_sets():
        return ["val"]

          

config = TimeBasedConfig(ts_ids=500, train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, features_to_take=['n_flows', 'n_packets'],
                        nan_threshold=0.5, random_state=1500, preprocess_order=["handling_anomalies", "filling_gaps", "transforming", PerFitTest])

time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(preprocess_order=["handling_anomalies", PerFitTest, "filling_gaps", "transforming"], workers=0)
# Or
time_based_dataset.set_preprocess_order(preprocess_order=["handling_anomalies", PerFitTest, "filling_gaps", "transforming"], workers=0)

```

### NoFitCustomHandler
- You can refer to [`NoFitCustomHandler`](reference_custom_handlers.md#cesnet_tszoo.utils.custom_handler.custom_handler.NoFitCustomHandler)
- One instance is created per time series
- Does not require nor supports fitting

```python

import numpy as np
from cesnet_tszoo.utils.custom_handler import NoFitCustomHandler
from cesnet_tszoo.configs import TimeBasedConfig

class NoFitTest(NoFitCustomHandler):
    def apply(self, data: np.ndarray) -> np.ndarray:
        data[:, :] = -1
        return data

    @staticmethod
    def get_target_sets():
        return ["test"]

          

config = TimeBasedConfig(ts_ids=500, train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, features_to_take=['n_flows', 'n_packets'],
                        nan_threshold=0.5, random_state=1500, preprocess_order=["handling_anomalies", "filling_gaps", "transforming", NoFitTest])

time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(preprocess_order=["handling_anomalies", NoFitTest, "filling_gaps", "transforming"], workers=0)
# Or
time_based_dataset.set_preprocess_order(preprocess_order=["handling_anomalies", NoFitTest, "filling_gaps", "transforming"], workers=0)

```