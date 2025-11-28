# Using anomaly handlers

This tutorial will look at some configuration options for using anomaly handlers.

Only time-based will be used, because all methods work almost the same way for other dataset types.

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`using_anomaly_handlers`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/using_anomaly_handlers.ipynb)

Relevant configuration values:

- `handle_anomalies_with` - Defines the anomaly handler used to transform anomalies in the train set.

## Anomaly handlers
- Anomaly handlers are implemented as class.
    - You can create your own or use built-in one.
- Every time series in train set has its own anomaly handler instance.
- Anomaly handler must implement `fit` and `transform_anomalies`.
- To use anomaly handler, train set must be implemented.
- Anomaly handler will only be used on train set.
- You can change used anomaly handler later with `update_dataset_config_and_initialize` or `apply_anomaly_handler`.

### Built-in
To see all built-in anomaly handlers refer to [`Anomaly handlers`](reference_anomaly_handlers.md#cesnet_tszoo.utils.anomaly_handler.anomaly_handler.ZScore).

```python

from cesnet_tszoo.utils.enums import AnomalyHandlerType
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=500, train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, features_to_take=['n_flows', 'n_packets'],
                         handle_anomalies_with=AnomalyHandlerType.Z_SCORE, nan_threshold=0.5, random_state=1500)                                                                           

# Call on time-based dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(handle_anomalies_with=AnomalyHandlerType.Z_SCORE, workers=0)
# Or
time_based_dataset.apply_anomaly_handler(handle_anomalies_with=AnomalyHandlerType.Z_SCORE, workers=0)

```

### Custom
You can create your own custom anomaly handler. It is recommended to derive from 'AnomalyHandler' base class. 

To check AnomalyHandler base class refer to [`AnomalyHandler`](reference_anomaly_handlers.md#cesnet_tszoo.utils.anomaly_handler.anomaly_handler.AnomalyHandler)

```python

import numpy as np
from cesnet_tszoo.utils.anomaly_handler import AnomalyHandler
from cesnet_tszoo.configs import TimeBasedConfig

class CustomAnomalyHandler(AnomalyHandler):
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None
        self.iqr = None

    def fit(self, data: np.ndarray) -> None:
        q25, q75 = np.percentile(data, [25, 75], axis=0)
        self.iqr = q75 - q25

        self.lower_bound = q25 - 1.5 * self.iqr
        self.upper_bound = q75 + 1.5 * self.iqr

    def transform_anomalies(self, data: np.ndarray) -> np.ndarray:
        mask_lower_outliers = data < self.lower_bound
        mask_upper_outliers = data > self.upper_bound

        data[mask_lower_outliers] = np.take(self.lower_bound, np.where(mask_lower_outliers)[1])
        data[mask_upper_outliers] = np.take(self.upper_bound, np.where(mask_upper_outliers)[1])              

config = TimeBasedConfig(ts_ids=500, train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, features_to_take=['n_flows', 'n_packets'],
                           handle_anomalies_with=CustomAnomalyHandler, nan_threshold=0.5, random_state=1500)                                                                    

time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(handle_anomalies_with=CustomAnomalyHandler, workers=0)
# Or
time_based_dataset.apply_anomaly_handler(handle_anomalies_with=CustomAnomalyHandler, workers=0)

```

### Changing when is anomaly handler applied
- You can change when is a anomaly handler applied with `preprocess_order` parameter

```python

from cesnet_tszoo.utils.utils.enums import AnomalyHandlerType
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=500, train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, features_to_take=['n_flows', 'n_packets'],
                           handle_anomalies_with=AnomalyHandlerType.Z_SCORE, nan_threshold=0.5, random_state=1500, preprocess_order=["handling_anomalies", "filling_gaps", "transforming"])
                                                                        
# Call on dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:


```python

time_based_dataset.update_dataset_config_and_initialize(preprocess_order=["filling_gaps", "handling_anomalies", "transforming"], workers=0)
# Or
time_based_dataset.set_preprocess_order(preprocess_order=["filling_gaps", "handling_anomalies", "transforming"], workers=0)

```