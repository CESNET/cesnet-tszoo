# Scalers

The `cesnet_tszoo` package supports various ways of using scalers to transform data. Scaler(s) can be created and fitted (on train set) when initializing dataset with config. Or already fitted scaler(s) can be passed to transform data.

### Built-in scalers
The `cesnet_tszoo` package comes with multiple built-in scalers. Not all of them support `partial_fit` though. To check built-in scalers refer to [`scalers`][cesnet_tszoo.utils.scaler].

### Custom scalers
It is possible to create and use own scalers. It is recommended to use prepared base class [`Scaler`][cesnet_tszoo.utils.scaler.Scaler].

## Using scalers on time-based dataset
Related config parameters in [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig]:

- `scale_with`:  Defines the scaler(s) to transform the dataset. Can pass enum [`ScalerType`][cesnet_tszoo.utils.enums.ScalerType] for built-in scaler, pass a type of custom scaler or instance of already fitted scaler(s).
- `create_scaler_per_time_series`: Whether to create a separate scaler for each time series or create one scaler for all time series.
- `partial_fit_initialized_scalers`: Whether to `partial_fit` already fitted scaler(s).

!!! warning "Time series in test_ts_ids"
    Time series in `test_ts_ids` will not be transformed when `create_scaler_per_time_series` = `True`. But they will be transformed when `create_scaler_per_time_series` = `False`.

!!! warning "fit vs partial_fit"
    When `create_scaler_per_time_series` = `True` and scalers are not pre-fitted, scalers must implement `fit` method. Else if you want to fit scalers, `partial_fit` method must be implemented. Check [`Scaler`][cesnet_tszoo.utils.scaler.Scaler] for details.

## Using scalers on series-based dataset
Series-based dataset always uses `create_scaler_per_time_series` = `False`.
Related config parameters in [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig]:

- `scale_with`:  Defines the scaler to transform the dataset. Can pass enum [`ScalerType`][cesnet_tszoo.utils.enums.ScalerType] for built-in scaler, pass a type of custom scaler or instance of already fitted scaler.
- `partial_fit_initialized_scalers`: Whether to `partial_fit` already fitted scaler.

!!! warning "partial_fit"
    Scaler must implement `partial_fit` method unless using already fitted scaler without fitting it on train data. Check [`Scaler`][cesnet_tszoo.utils.scaler.Scaler] for details.    