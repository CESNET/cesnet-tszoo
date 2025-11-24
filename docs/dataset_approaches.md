The `cesnet-tszoo` library provides multiple splitting strategies to accommodate different TSA tasks, such as forecasting, classification, and similarity search.

## [`Time-based`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset)
Time-based approach splits each time series separately based on the time axis. The times of splits into train, validation, and test sets can be selected in multiple ways, for example, exact timestamp or classical percentage split (i.e., 60:20:20). The train set is always before the validation and test set, and the validation set is always before the test set. This splitting approach is practical, for example, for forecasting or anomaly detection, where we need to predict future data from historical data.
![ts_zoo-time_based.svg](./images/ts_zoo-time_based.svg)

## [`Time-based splitting with disjoint identifiers`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset)
Time-based splitting with disjoint identifiers was implemented to support better generalization of algorithms. Time series are split into train, validation, and test sets not only by time but simultaneously by identifiers. This approach allows robust evaluation of a model trained and validated on different time series in a different time span. 
![ts_zoo-disjoint_time_based.svg](./images/ts_zoo-disjoint_time_based.svg)

## [`Series-based`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset)
Series-based splitting procedure splits time series based on the different time series identifiers into train, validation, and test sets. The series-based splitting is valuable, for example, for classification based on time series behavior or similarity detection in the same time frame.
![ts_zoo-series_based.svg](./images/ts_zoo-series_based.svg)
