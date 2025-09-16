# Anomaly handlers

The `cesnet_tszoo` package supports various ways of using anomaly handlers to handle anomalies. Anomaly handlers can be created and fitted (on train set) when initializing dataset with config and each time series has its own anomaly handler instance.

Possible config parameters in [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig], [`DisjointTimeBasedConfig`][cesnet_tszoo.configs.disjoint_time_based_config.DisjointTimeBasedConfig] and [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig]:

- `handle_anomalies_with`:  Defines the anomaly handlers to handle anomalies in the train set. Can pass enum [`AnomalyHandlerType`][cesnet_tszoo.utils.enums.AnomalyHandlerType] for built-in anomaly handler or pass a type of custom anomaly handler.

### Built-in anomaly handlers
The `cesnet_tszoo` package comes with multiple built-in anomaly handlers. To check built-in anomaly handlers refer to [`anomaly handlers`][cesnet_tszoo.utils.anomaly_handler].

### Custom anomaly handlers
It is possible to create and use own anomaly handlers. It is recommended to use prepared base class [`AnomalyHandler`][cesnet_tszoo.utils.anomaly_handler.AnomalyHandler].