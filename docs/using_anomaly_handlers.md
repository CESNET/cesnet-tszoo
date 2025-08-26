# Using anomaly handlers

This tutorial will look at some configuration options for using anomaly handlers.

Each dataset type will have its own part because of multiple differences of available configuration values.

## [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`time_based_using_anomaly_handlers`](https://github.com/CESNET/cesnet-tszoo/blob/main/tutorial_notebooks/time_based_using_anomaly_handlers.ipynb)

Relevant configuration values:

- `handle_anomalies_with` - Defines the anomaly handler used to transform anomalies the dataset.

### Anomaly handlers
- Anomaly handlers are implemented as class.
    - You can create your own or use built-in one.
- Anomaly handler is applied before `default_values` and fillers took care of missing values.
- Every time series has its own anomaly handler instance.
- Anomaly handler must implement `fit` and `transform_anomalies`.
- To use anomaly handler, train set must be implemented.
- Anomaly handler will be fitted on train set.
- You can change used anomaly handler later with `update_dataset_config_and_initialize` or `apply_anomaly_handler`.

#### Built-in
To see all built-in anomaly handlers refer to [`Anomaly handlers`][cesnet_tszoo.utils.anomaly_handler.ZScore].

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

#### Custom
You can create your own custom anomaly handler. It is recommended to derive from 'AnomalyHandler' base class. 

To check AnomalyHandler base class refer to [`AnomalyHandler`][cesnet_tszoo.utils.anomaly_handler.AnomalyHandler]

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

## [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`disjoint_time_based_using_anomaly_handlers`](https://github.com/CESNET/cesnet-tszoo/blob/main/tutorial_notebooks/disjoint_time_based_using_anomaly_handlers.ipynb)

Relevant configuration values:

- `handle_anomalies_with` - Defines the anomaly handler used to transform anomalies the dataset.

### Transformers
- Anomaly handlers are implemented as class.
    - You can create your own or use built-in one.
- Anomaly handler is applied before `default_values` and fillers took care of missing values.
- Every time series has its own anomaly handler instance.
- Anomaly handler must implement `fit` and `transform_anomalies`.
- To use anomaly handler, train set must be implemented.
- Anomaly handler will only be used on train set.
- You can change used anomaly handler later with `update_dataset_config_and_initialize` or `apply_anomaly_handler`.

#### Built-in
To see all built-in anomaly handlers refer to [`Anomaly handlers`][cesnet_tszoo.utils.anomaly_handler.ZScore].

```python

from cesnet_tszoo.utils.enums import AnomalyHandlerType
from cesnet_tszoo.configs import DisjointTimeBasedConfig

config = DisjointTimeBasedConfig(train_ts=500, val_ts=None, test_ts=None, train_time_period=0.5, features_to_take=["n_flows", "n_packets"],
                                 handle_anomalies_with=AnomalyHandlerType.Z_SCORE, nan_threshold=0.5, random_state=1500)                                                                      

# Call on disjoint-time-based dataset to use created config
disjoint_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

disjoint_dataset.update_dataset_config_and_initialize(handle_anomalies_with=AnomalyHandlerType.Z_SCORE, workers=0)
# Or
disjoint_dataset.apply_anomaly_handler(handle_anomalies_with=AnomalyHandlerType.Z_SCORE, workers=0)

```

#### Custom
You can create your own custom anomaly handler. It is recommended to derive from 'AnomalyHandler' base class. 

To check AnomalyHandler base class refer to [`AnomalyHandler`][cesnet_tszoo.utils.anomaly_handler.AnomalyHandler]

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

config = DisjointTimeBasedConfig(train_ts=500, val_ts=None, test_ts=None, train_time_period=0.5, features_to_take=["n_flows", "n_packets"],
                                 handle_anomalies_with=CustomAnomalyHandler, nan_threshold=0, random_state=1500)                                                                    

disjoint_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

disjoint_dataset.update_dataset_config_and_initialize(handle_anomalies_with=CustomAnomalyHandler, workers=0)
# Or
disjoint_dataset.apply_anomaly_handler(handle_anomalies_with=CustomAnomalyHandler, workers=0)

```

## [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`series_based_using_anomaly_handlers`](https://github.com/CESNET/cesnet-tszoo/blob/main/tutorial_notebooks/series_based_using_anomaly_handlers.ipynb)

Relevant configuration values:

- `handle_anomalies_with` - Defines the anomaly handler used to transform anomalies the dataset.

### Transformers
- Anomaly handlers are implemented as class.
    - You can create your own or use built-in one.
- Anomaly handler is applied before `default_values` and fillers took care of missing values.
- Every time series has its own anomaly handler instance.
- Anomaly handler must implement `fit` and `transform_anomalies`.
- To use anomaly handler, train set must be implemented.
- Anomaly handler will only be used on train set.
- You can change used anomaly handler later with `update_dataset_config_and_initialize` or `apply_anomaly_handler`.

#### Built-in
To see all built-in anomaly handlers refer to [`Anomaly handlers`][cesnet_tszoo.utils.anomaly_handler.ZScore].

```python

from cesnet_tszoo.utils.enums import AnomalyHandlerType
from cesnet_tszoo.configs import SeriesBasedConfig

config = SeriesBasedConfig(time_period=0.5, train_ts=500, features_to_take=["n_flows", "n_packets"],
                           handle_anomalies_with=AnomalyHandlerType.Z_SCORE, nan_threshold=0.5, random_state=1500)                                                                    

# Call on series-based dataset to use created config
series_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

series_based_dataset.update_dataset_config_and_initialize(handle_anomalies_with=AnomalyHandlerType.Z_SCORE, workers=0)
# Or
series_based_dataset.apply_anomaly_handler(handle_anomalies_with=AnomalyHandlerType.Z_SCORE, workers=0)

```

#### Custom
You can create your own custom anomaly handler. It is recommended to derive from 'AnomalyHandler' base class. 

To check AnomalyHandler base class refer to [`AnomalyHandler`][cesnet_tszoo.utils.anomaly_handler.AnomalyHandler]

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

config = SeriesBasedConfig(time_period=0.5, train_ts=500, features_to_take=["n_flows", "n_packets"],
                           handle_anomalies_with=CustomAnomalyHandler, nan_threshold=0, random_state=1500)                                                                 

series_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

series_based_dataset.update_dataset_config_and_initialize(handle_anomalies_with=CustomAnomalyHandler, workers=0)
# Or
series_based_dataset.apply_anomaly_handler(handle_anomalies_with=CustomAnomalyHandler, workers=0)

```   