# Using transformers

This tutorial will look at some configuration options for using transformers.

Each dataset type will have its own part because of multiple differences of available configuration values.

## [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`time_based_using_transformers`](https://github.com/CESNET/cesnet-tszoo/blob/main/tutorial_notebooks/time_based_using_transformers.ipynb)

Relevant configuration values:

- `transform_with` - Defines the transformer used to transform the dataset.
- `create_transformer_per_time_series` - If True, a separate transformer is created for each time series and transformers wont be used for time series on 'test_ts_id'.
- `partial_fit_initialized_transformers` - If True, partial fitting on train set is performed when using initiliazed transformers.

### Transformers
- Transformers are implemented as class.
    - You can create your own or use built-in one.
- Transformer must implement `transform`.
- Transformers are applied after `default_values` and fillers took care of missing values.
- To use transformers, train set must be implemented (unless transformers are already fitted and `partial_fit_initialized_transformers` is False).
- `fit` method on transformer:
    - must be implemented when `create_transformer_per_time_series` is True and transformers are not already fitted.
- `partial_fit` method on transformer:
    - must be implemented when `create_transformer_per_time_series` is False or using already fitted transformers with `partial_fit_initialized_transformers` set to True.
- You can change used transformer later with `update_dataset_config_and_initialize` or `apply_transformer`.

#### Built-in
To see all built-in transformers refer to [`Transformers`][cesnet_tszoo.utils.transformer.LogTransformer].

```python

from cesnet_tszoo.utils.enums import TransformerType
from cesnet_tszoo.configs import TimeBasedConfig

config = TimeBasedConfig(ts_ids=[1367, 1368], train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, test_ts_ids=[1370], features_to_take=['n_flows', 'n_packets'],
                         transform_with=TransformerType.MIN_MAX_SCALER, create_transformer_per_time_series=True)                                                                              

# Call on time-based dataset to use created config
time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(transform_with=TransformerType.MIN_MAX_SCALER, create_transformer_per_time_series=True, partial_fit_initialized_transformers="config", workers=0)
# Or
time_based_dataset.apply_transformer(transform_with=TransformerType.MIN_MAX_SCALER, create_transformer_per_time_series=True, partial_fit_initialized_transformers="config", workers=0)

```

#### Custom
You can create your own custom transformer. It is recommended to derive from 'Transformer' base class. 

To check Transformer base class refer to [`Transformer`][cesnet_tszoo.utils.transformer.Transformer]

```python

from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.configs import TimeBasedConfig

class CustomTransformer(Transformer):
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
                         transform_with=CustomTransformer, create_transformer_per_time_series=True)                                                                        

time_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

time_based_dataset.update_dataset_config_and_initialize(transform_with=CustomTransformer, create_transformer_per_time_series=True, partial_fit_initialized_transformers="config", workers=0)
# Or
time_based_dataset.apply_transformer(transform_with=CustomTransformer, create_transformer_per_time_series=True, partial_fit_initialized_transformers="config", workers=0)

```

#### Using already fitted transformers

```python

from cesnet_tszoo.configs import TimeBasedConfig         

config = TimeBasedConfig(ts_ids=[103, 118], train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, test_ts_ids=[1370], features_to_take=['n_flows', 'n_packets'],
                         transform_with=list_of_fitted_transformers, create_transformer_per_time_series=True)    

# Length of list_of_fitted_transformers must be equal to number of time series in ts_ids 
# All transformers in list_of_fitted_transformers must be of same type                                                            

config = TimeBasedConfig(ts_ids=[103, 118], train_time_period=0.5, val_time_period=0.2, test_time_period=0.1, test_ts_ids=[1370], features_to_take=['n_flows', 'n_packets'],
                         transform_with=one_prefitted_transformer, create_transformer_per_time_series=True)

# one_prefitted_transformer must be just one transformer (not a list)                     

time_based_dataset.set_dataset_config_and_initialize(config)

```

## [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] dataset

!!! info "Note"
    For every configuration and more detailed examples refer to Jupyter notebook [`series_based_using_transformers`](https://github.com/CESNET/cesnet-tszoo/blob/main/tutorial_notebooks/series_based_using_transformers.ipynb)

Relevant configuration values:

- `transform_with` - Defines the transformer used to transform the dataset.
- `partial_fit_initialized_transformers` - If True, partial fitting on train set is performed when using initiliazed transformer.

### Transformers
- Transformers are implemented as class.
    - You can create your own or use built-in one.
- Transformer is applied after `default_values` and fillers took care of missing values.
- One transformer is used for all time series.
- Transformer must implement `transform`.
- Transformer must implement `partial_fit` (unless transformer is already fitted and `partial_fit_initialized_transformers` is False).
- To use transformer, train set must be implemented (unless transformer is already fitted and `partial_fit_initialized_transformers` is False).
- You can change used transformer later with `update_dataset_config_and_initialize` or `apply_transformer`.

#### Built-in
To see all built-in transformers refer to [`Transformers`][cesnet_tszoo.utils.transformer.LogTransformer].

```python

from cesnet_tszoo.utils.enums import TransformerType
from cesnet_tszoo.configs import SeriesBasedConfig

config = SeriesBasedConfig(time_period=0.5, train_ts=500, features_to_take=["n_flows", "n_packets"],
                           transform_with=TransformerType.MIN_MAX_SCALER, nan_threshold=0.5, random_state=1500)                                                                          

# Call on series-based dataset to use created config
series_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

series_based_dataset.update_dataset_config_and_initialize(transform_with=TransformerType.MIN_MAX_SCALER, partial_fit_initialized_transformers="config", workers=0)
# Or
series_based_dataset.apply_transformer(transform_with=TransformerType.MIN_MAX_SCALER, partial_fit_initialized_transformers="config", workers=0)

```

#### Custom
You can create your own custom transformer. It is recommended to derive from 'Transformer' base class. 

To check Transformer base class refer to [`Transformer`][cesnet_tszoo.utils.transformer.Transformer]

```python

from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.configs import SeriesBasedConfig

class CustomTransformer(Transformer):
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
                           transform_with=CustomTransformer, nan_threshold=0.5, random_state=1500)                                                                    

series_based_dataset.set_dataset_config_and_initialize(config)

```

Or later with:

```python

series_based_dataset.update_dataset_config_and_initialize(transform_with=CustomTransformer, partial_fit_initialized_transformers="config", workers=0)
# Or
series_based_dataset.apply_transformer(transform_with=CustomTransformer, partial_fit_initialized_transformers="config", workers=0)

```

#### Using already fitted transformers

```python

from cesnet_tszoo.configs import SeriesBasedConfig         

config = SeriesBasedConfig(time_period=0.5, val_ts=500, features_to_take=["n_flows", "n_packets"],
                           transform_with=fitted_transformer, nan_threshold=0.5, random_state=999)   

# fitted_transformer must be just one transformer (not a list)                                                                     

series_based_dataset.set_dataset_config_and_initialize(config)

```    