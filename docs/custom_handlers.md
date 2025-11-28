# Custom handlers {#cesnet_tszoo.general.custom_handlers}

The `cesnet_tszoo` package supports adding custom way of handling data. It is possible to create custom handler for per time series, custom handler for all time series by subclassing from [`custom handlers`](reference_custom_handlers.md#cesnet_tszoo.utils.custom_handler.custom_handler). They also support applying only to specified sets.

Related config parameters in [`TimeBasedConfig`](reference_time_based_config.md#references.TimeBasedConfig), [`DisjointTimeBasedConfig`](reference_disjoint_time_based_config.md#references.DisjointTimeBasedConfig) and [`SeriesBasedConfig`](reference_series_based_config.md#references.SeriesBasedConfig):

- `preprocess_order`: Mainly used for changing order of preprocesses. Is also used as a way of adding custom handlers by adding type their type between the preprocesses.

## Creating custom handlers
You can create custom handler by subclassing from one of the below classes:

- [`PerSeriesCustomHandler`](reference_custom_handlers.md#cesnet_tszoo.utils.custom_handler.custom_handler.PerSeriesCustomHandler)
    - Instance is created for every time series separately
    - Fits on their respective time series train set part
- [`AllSeriesCustomHandler`](reference_custom_handlers.md#cesnet_tszoo.utils.custom_handler.custom_handler.AllSeriesCustomHandler) 
    - Only one instance is created for all used time series
    - Fits on train set
- [`NoFitCustomHandler`](reference_custom_handlers.md#cesnet_tszoo.utils.custom_handler.custom_handler.NoFitCustomHandler) 
    - Instance is created for every time series separately 
    - Does not fit

!!! warning "PerSeriesCustomHandler"
    [`PerSeriesCustomHandler`](reference_custom_handlers.md#cesnet_tszoo.utils.custom_handler.custom_handler.PerSeriesCustomHandler) is only supported for [`Time-based`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), because of its nature.