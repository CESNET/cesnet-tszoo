# Using scalers

This tutorial will look at some configuration options for using scalers.

Each dataset type will have its own part because of multiple differences of available configuration values.

## [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`time_based_using_scalers`](https://github.com/CESNET/cesnet-tszoo/blob/tutorial_notebooks/time_based_using_scalers.ipynb)

Relevant configuration values:

- `scale_with` - Defines the scaler used to transform the dataset.
- `create_scaler_per_time_series` - If True, a separate scaler is created for each time series and scalers wont be used for time series on 'test_ts_id'.
- `partial_fit_initialized_scalers` - If True, partial fitting on train set is performed when using initiliazed scalers.

### Scalers
- Scalers are implemented as class.
    - You can create your own or use built-in one.
- Scaler must implement `transform`.
- Scalers are applied after `default_values` and fillers took care of missing values.
- To use scalers, train set must be implemented (unless scalers are already fitted and `partial_fit_initialized_scalers` is False).
- `fit` method on scaler:
    - must be implemented when `create_scaler_per_time_series` is True and scalers are not already fitted.
- `partial_fit` method on scaler:
    - must be implemented when `create_scaler_per_time_series` is False or using already fitted scalers with `partial_fit_initialized_scalers` set to True.
- You can change used scaler later with `update_dataset_config_and_initialize` or `apply_scaler`.

#### Built-in
To see all built-in scalers refer to [`Scalers`][cesnet_tszoo.utils.scaler.LogScaler].

```python

from cesnet_tszoo.utils.enums import ScalerType
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=[1367, 1368], train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, test_ts_ids=[1370], features_to_take=['n_flows', 'n_packets'],
                         scale_with=ScalerType.MIN_MAX_SCALER, create_scaler_per_time_series=True)                                                                              

# Call on time-based dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(scale_with=ScalerType.MIN_MAX_SCALER, create_scaler_per_time_series=True, partial_fit_initialized_scalers="config", workers=0)
# Or
time_based_dataset.apply_scaler(scale_with=ScalerType.MIN_MAX_SCALER, create_scaler_per_time_series=True, partial_fit_initialized_scalers="config", workers=0)

```

#### Custom
You can create your own custom scaler. It is recommended to derive from 'Scaler' base class. 

To check Scaler base class refer to [`Scaler`][cesnet_tszoo.utils.scaler.Scaler]

```python

from cesnet_tszoo.utils.scaler import Scaler
from cesnet_tszoo.configs import TimeBasedConfig

class CustomScaler(Scaler):
    def __init__(self):
        super().__init__()
        
        self.max = None
        self.min = None
    
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)
    
    def fit(self, data):
        self.partial_fit(data)
    
    def partial_fit(self, data):
        
        if self.max is None and self.min is None:
            self.max = np.max(data, axis=0)
            self.min = np.min(data, axis=0)
            return
        
        temp_max = np.max(data, axis=0)
        temp = np.vstack((self.max, temp_max)) 
        self.max = np.max(temp, axis=0)
        
        temp_min = np.min(data, axis=0)
        temp = np.vstack((self.min, temp_min)) 
        self.min = np.min(temp, axis=0)            

config = TimeBasedConfig(ts_ids=[1367, 1368], train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, test_ts_ids=[1370], features_to_take=['n_flows', 'n_packets'],
                         scale_with=CustomScaler, create_scaler_per_time_series=True)                                                                        

time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(scale_with=CustomScaler, create_scaler_per_time_series=True, partial_fit_initialized_scalers="config", workers=0)
# Or
time_based_dataset.apply_scaler(scale_with=CustomScaler, create_scaler_per_time_series=True, partial_fit_initialized_scalers="config", workers=0)

```

#### Using already fitted scalers

```python

from cesnet_tszoo.configs import TimeBasedConfig         

config = TimeBasedConfig(ts_ids=[103, 118], train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, test_ts_ids=[1370], features_to_take=['n_flows', 'n_packets'],
                         scale_with=list_of_fitted_scalers, create_scaler_per_time_series=True)    

# Length of list_of_fitted_scalers must be equal to number of time series in ts_ids 
# All scalers in list_of_fitted_scalers must be of same type                                                            

config = TimeBasedConfig(ts_ids=[103, 118], train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, test_ts_ids=[1370], features_to_take=['n_flows', 'n_packets'],
                         scale_with=one_prefitted_scaler, create_scaler_per_time_series=True)

# one_prefitted_scaler must be just one scaler (not a list)                     

time_based_dataset.set_dataset_config_and_initialize(config)

```

## [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`series_based_using_scalers`](https://github.com/CESNET/cesnet-tszoo/blob/tutorial_notebooks/series_based_using_scalers.ipynb)

Relevant configuration values:

- `scale_with` - Defines the scaler used to transform the dataset.
- `partial_fit_initialized_scalers` - If True, partial fitting on train set is performed when using initiliazed scaler.

### Scalers
- Scalers are implemented as class.
    - You can create your own or use built-in one.
- Scaler is applied after `default_values` and fillers took care of missing values.
- One scaler is used for all time series.
- Scaler must implement `transform`.
- Scaler must implement `partial_fit` (unless scaler is already fitted and `partial_fit_initialized_scalers` is False).
- To use scaler, train set must be implemented (unless scaler is already fitted and `partial_fit_initialized_scalers` is False).
- You can change used scaler later with `update_dataset_config_and_initialize` or `apply_scaler`.

#### Built-in
To see all built-in scalers refer to [`Scalers`][cesnet_tszoo.utils.scaler.LogScaler].

```python

from cesnet_tszoo.utils.enums import ScalerType
from cesnet_tszoo.configs import SeriesBasedConfig

config = SeriesBasedConfig(time_period=0.5, train_ts=500, features_to_take=["n_flows", "n_packets"],
                           scale_with=ScalerType.MIN_MAX_SCALER, nan_threshold=0.5, random_state=1500)                                                                          

# Call on series-based dataset to use created config
series_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

series_based_dataset.update_dataset_config_and_initialize(scale_with=ScalerType.MIN_MAX_SCALER, partial_fit_initialized_scalers="config", workers=0)
# Or
series_based_dataset.apply_scaler(scale_with=ScalerType.MIN_MAX_SCALER, partial_fit_initialized_scalers="config", workers=0)

```

#### Custom
You can create your own custom scaler. It is recommended to derive from 'Scaler' base class. 

To check Scaler base class refer to [`Scaler`][cesnet_tszoo.utils.scaler.Scaler]

```python

from cesnet_tszoo.utils.scaler import Scaler
from cesnet_tszoo.configs import SeriesBasedConfig

class CustomScaler(Scaler):
    def __init__(self):
        super().__init__()
        
        self.max = None
        self.min = None
    
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)
    
    def fit(self, data):
        self.partial_fit(data)
    
    def partial_fit(self, data):
        
        if self.max is None and self.min is None:
            self.max = np.max(data, axis=0)
            self.min = np.min(data, axis=0)
            return
        
        temp_max = np.max(data, axis=0)
        temp = np.vstack((self.max, temp_max)) 
        self.max = np.max(temp, axis=0)
        
        temp_min = np.min(data, axis=0)
        temp = np.vstack((self.min, temp_min)) 
        self.min = np.min(temp, axis=0)            

config = SeriesBasedConfig(time_period=0.5, train_ts=500, features_to_take=["n_flows", "n_packets"],
                           scale_with=CustomScaler, nan_threshold=0.5, random_state=1500)                                                                    

series_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

series_based_dataset.update_dataset_config_and_initialize(scale_with=CustomScaler, partial_fit_initialized_scalers="config", workers=0)
# Or
series_based_dataset.apply_scaler(scale_with=CustomScaler, partial_fit_initialized_scalers="config", workers=0)

```

#### Using already fitted scalers

```python

from cesnet_tszoo.configs import SeriesBasedConfig         

config = SeriesBasedConfig(time_period=0.5, val_ts=500, features_to_take=["n_flows", "n_packets"],
                           scale_with=fitted_scaler, nan_threshold=0.5, random_state=999)   

# fitted_scaler must be just one scaler (not a list)                                                                     

series_based_dataset.set_dataset_config_and_initialize(config)

```    