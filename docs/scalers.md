# Transformers

The `cesnet_tszoo` package supports various ways of using transformers to transform data. Transformer(s) can be created and fitted (on train set) when initializing dataset with config. Or already fitted transformer(s) can be passed to transform data.

### Built-in transformers
The `cesnet_tszoo` package comes with multiple built-in transformers. Not all of them support `partial_fit` though. To check built-in transformers refer to [`transformers`][cesnet_tszoo.utils.transformer].

### Custom transformers
It is possible to create and use own transformers. It is recommended to use prepared base class [`Transformer`][cesnet_tszoo.utils.transformer.Transformer].

## Using transformers on time-based dataset
Related config parameters in [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig]:

- `transform_with`:  Defines the transformer(s) to transform the dataset. Can pass enum [`TransformerType`][cesnet_tszoo.utils.enums.TransformerType] for built-in transformer, pass a type of custom transformer or instance of already fitted transformer(s).
- `create_transformer_per_time_series`: Whether to create a separate transformer for each time series or create one transformer for all time series.
- `partial_fit_initialized_transformers`: Whether to `partial_fit` already fitted transformer(s).

!!! warning "Time series in test_ts_ids"
    Time series in `test_ts_ids` will not be transformed when `create_transformer_per_time_series` = `True`. But they will be transformed when `create_transformer_per_time_series` = `False`.

!!! warning "fit vs partial_fit"
    When `create_transformer_per_time_series` = `True` and transformers are not pre-fitted, transformers must implement `fit` method. Else if you want to fit transformers, `partial_fit` method must be implemented. Check [`Transformer`][cesnet_tszoo.utils.transformer.Transformer] for details.

## Using transformers on series-based dataset
Series-based dataset always uses `create_transformer_per_time_series` = `False`.
Related config parameters in [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig]:

- `transform_with`:  Defines the transformer to transform the dataset. Can pass enum [`TransformerType`][cesnet_tszoo.utils.enums.TransformerType] for built-in transformer, pass a type of custom transformer or instance of already fitted transformer.
- `partial_fit_initialized_transformers`: Whether to `partial_fit` already fitted transformer.

!!! warning "partial_fit"
    Transformer must implement `partial_fit` method unless using already fitted transformer without fitting it on train data. Check [`Transformer`][cesnet_tszoo.utils.transformer.Transformer] for details.    