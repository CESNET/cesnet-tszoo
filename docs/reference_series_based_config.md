# Series-based config class {#cesnet_tszoo.references.configs.SeriesBasedConfig}

This class is used for configuring the [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset].

Used to configure the following:

- Train, validation, test, all sets (time period, sizes, features)
- Handling missing values (default values, [`fillers`][cesnet_tszoo.utils.filler.filler])
- Handling anomalies ([`anomaly handlers`][cesnet_tszoo.utils.anomaly_handler.anomaly_handler])
- Data transformation using [`transformers`][cesnet_tszoo.utils.transformer.transformer]
- Applying custom handlers ([`custom handlers`][cesnet_tszoo.utils.custom_handler.custom_handler])
- Changing order of preprocesses
- Dataloader options (train/val/test/all/init workers, batch size, train loading order)
- Plotting

**Important Notes:**

- Custom fillers must inherit from the [`fillers`][cesnet_tszoo.utils.filler.filler.Filler] base class.
- Custom anomaly handlers must inherit from the [`anomaly handlers`][cesnet_tszoo.utils.anomaly_handler.anomaly_handler.AnomalyHandler] base class.
- Selected anomaly handler is only used for train set.    
- It is recommended to use the [`transformers`][cesnet_tszoo.utils.transformer.transformer.Transformer] base class, though this is not mandatory as long as it meets the required methods.
    - If a transformer is already initialized and `partial_fit_initialized_transformers` is `False`, the transformer does not require `partial_fit`.
    - Otherwise, the transformer must support `partial_fit`.
    - Transformers must implement `transform` method.
    - Both `partial_fit` and `transform` methods must accept an input of type `np.ndarray` with shape `(times, features)`.
- Custom handlers must be derived from one of the built-in [`custom handler`][cesnet_tszoo.utils.custom_handler.custom_handler] classes 
- `train_ts`, `val_ts`, and `test_ts` must not contain any overlapping time series IDs.

::: cesnet_tszoo.configs.series_based_config.SeriesBasedConfig
    options:
      show_root_heading: false
      show_source: true
      show_docstring: false
      show_signature: false
      show_bases: false
      members: false
      show_docstring_attributes: false
      show_docstring_classes: false
      show_docstring_description: false
      show_root_toc_entry: false
      show_category_heading: false
      parameter_headings: false
      group_by_category: false

## Configuration options

| Name                               | Default                                         | Description                                                                                                                                                                                                                          |
|------------------------------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| time_period                        |                                                 | Defines the time period for returning data from `train/val/test/all`. Can be a range of time IDs, a tuple of datetime objects or a float. Float value is equivalent to percentage of available times from start.                     |
| train_ts                            | None                                            | Defines which time series IDs are used in the training set. Can be a list of IDs, or an integer/float to specify a random selection. An `int` specifies the number of random time series, and a `float` specifies the proportion of available time series. `int` and `float` must be greater than 0, and a float should be smaller or equal to 1.0. Using `int` or `float` guarantees that no time series from other sets will be used. |
| val_ts                              | None                                            | Defines which time series IDs are used in the validation set. Same as `train_ts` but for the validation set.                                                                                                                        |
| test_ts                             | None                                            | Defines which time series IDs are used in the test set. Same as `train_ts` but for the test set.                                                                                                                                     |
| features_to_take                    | "all"                                           | Defines which features are used.                                                                                                                                                                                                     |
| default_values                      | "default"                                       | Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature.                                                                                                            |
| train_batch_size                    | 32                                              | Batch size for the train dataloader. Affects number of returned time series in one batch.                                                                                                                                            |
| val_batch_size                      | 64                                              | Batch size for the validation dataloader. Affects number of returned time series in one batch.                                                                                                                                       |
| test_batch_size                     | 128                                             | Batch size for the test dataloader. Affects number of returned time series in one batch.                                                                                                                                             |
| all_batch_size                       | 128                                             | Batch size for the all dataloader. Affects number of returned time series in one batch.                                                                                                                                              |
| preprocess_order                    | ["handling_anomalies", "filling_gaps", "transforming"] | Defines in which order preprocesses are used. Also can add to order a type of `AllSeriesCustomHandler` or `NoFitCustomHandler`.                                                                                                     |
| fill_missing_with                   | None                                            | Defines how to fill missing values in the dataset. Can pass enum `FillerType` for built-in filler or pass a type of custom filler that must derive from `Filler` base class.                                                        |
| transform_with                      | None                                            | Defines the transformer used to transform the dataset. Can pass enum `TransformerType` for built-in transformer, pass a type of custom transformer or instance of already fitted transformer.                                   |
| handle_anomalies_with               | None                                            | Defines the anomaly handler for handling anomalies in the train set. Can pass enum `AnomalyHandlerType` for built-in anomaly handler or a type of custom anomaly handler.                                                            |
| partial_fit_initialized_transformer | False                                           | If `True`, partial fitting on train set is performed when using initiliazed transformer.                                                                                                                                            |
| include_time                        | True                                            | If `True`, time data is included in the returned values.                                                                                                                                                                              |
| include_ts_id                       | True                                            | If `True`, time series IDs are included in the returned values.                                                                                                                                                                       |
| time_format                         | TimeFormat.ID_TIME                              | Format for the returned time data. When using TimeFormat.DATETIME, time will be returned as separate list along rest of the values.                                                                                                  |
| train_workers                       | 4                                               | Number of workers for loading training data. `0` means that the data will be loaded in the main process.                                                                                                                             |
| val_workers                         | 3                                               | Number of workers for loading validation data. `0` means that the data will be loaded in the main process.                                                                                                                            |
| test_workers                        | 2                                               | Number of workers for loading test data. `0` means that the data will be loaded in the main process.                                                                                                                                 |
| all_workers                          | 4                                               | Number of workers for loading all data. `0` means that the data will be loaded in the main process.                                                                                                                                  |
| init_workers                        | 4                                               | Number of workers for initial dataset processing during configuration. `0` means that the data will be loaded in the main process.                                                                                                   |
| nan_threshold                        | 1.0                                             | Maximum allowable percentage of missing data. Time series exceeding this threshold are excluded. Time series over the threshold will not be used. Used for `train/val/test/all` separately.                                          |
| train_dataloader_order              | DataloaderOrder.SEQUENTIAL                       | Defines the order of data returned by the training dataloader.                                                                                                                                                                        |
| random_state                        | None                                            | Fixes randomness for reproducibility during configuration and dataset initialization.                                                                                                                                                 |

## Various attributes
Various attributes that are used for the inner workings of the config.

**Attributes:**

| Name                                | Type       | Description                                                                                                                                                          |
|-------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| used_train_workers                  | int       | Tracks the number of train workers in use. Helps determine if the train dataloader should be recreated based on worker changes.                                      |
| used_val_workers                    | int       | Tracks the number of validation workers in use. Helps determine if the validation dataloader should be recreated based on worker changes.                             |
| used_test_workers                   | int       | Tracks the number of test workers in use. Helps determine if the test dataloader should be recreated based on worker changes.                                         |
| used_all_workers                    | int       | Tracks the total number of all workers in use. Helps determine if the all dataloader should be recreated based on worker changes.                                      |
| uses_all_ts                         | bool      | Whether all time series set should be used.                                                                                                                          |
| import_identifier                   | str | Tracks the name of the config upon import. None if not imported.                                                                                                     |
| filler_factory                       | FillerFactory      | Represents factory used to create passed Filler type.                                                                                                                |
| anomaly_handler_factory              | AnomalyHandlerFactory      | Represents factory used to create passed Anomaly Handler type.                                                                                                       |
| transformer_factory                  | TransformerFactory      | Represents factory used to create passed Transformer type.                                                                                                           |
| can_fit_fillers                       | bool      | Whether fillers in this config can be fitted.                                                                                                                        |
| logger                               | Logger    | Logger for displaying information.  

The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize] is called:

**Attributes:**

| Name                                | Type       | Description                                                                                                                                                          |
|-------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| all_ts                               | np.ndarray      | If no specific sets (train/val/test) are provided, all time series IDs are used. When any set is defined, only the time series IDs in defined sets are used.        |
| train_ts_row_ranges                  | np.ndarray      | Initialized when `train_ts` is set. Contains time series IDs in train set with their respective time ID ranges.                                                     |
| val_ts_row_ranges                    | np.ndarray      | Initialized when `val_ts` is set. Contains time series IDs in validation set with their respective time ID ranges.                                                  |
| test_ts_row_ranges                   | np.ndarray      | Initialized when `test_ts` is set. Contains time series IDs in test set with their respective time ID ranges.                                                       |
| all_ts_row_ranges                    | np.ndarray      | Initialized when `all_ts` is set. Contains time series IDs in all set with their respective time ID ranges.                                                         |
| display_time_period                  | range       | Used to display the configured value of `time_period`.                                                                                                              |
| aggregation                          | AggregationType | The aggregation period used for the data.                                                                                                                          |
| source_type                          | SourceType       | The source type of the data.                                                                                                                                         |
| database_name                         | str       | Specifies which database this config applies to.                                                                                                                    |
| features_to_take_without_ids         | np.ndarray      | Features to be returned, excluding time or time series IDs.                                                                                                         |
| indices_of_features_to_take_no_ids   | np.ndarray      | Indices of non-ID features in `features_to_take`.                                                                                                                   |
| ts_id_name                            | str       | Name of the time series ID, dependent on `source_type`.                                                                                                             |
| used_singular_train_time_series      | int      | Currently used singular train set time series for dataloader.                                                                                                       |
| used_singular_val_time_series        | int      | Currently used singular validation set time series for dataloader.                                                                                                  |
| used_singular_test_time_series       | int      | Currently used singular test set time series for dataloader.                                                                                                        |
| used_singular_all_time_series        | int      | Currently used singular all set time series for dataloader.                                                                                                         |
| train_preprocess_order                | np.ndarray      | All preprocesses used for train set.                                                                                                                               |
| val_preprocess_order                  | np.ndarray      | All preprocesses used for val set.                                                                                                                                |
| test_preprocess_order                 | np.ndarray      | All preprocesses used for test set.                                                                                                                               |
| all_preprocess_order                  | np.ndarray      | All preprocesses used for all set.                                                                                                                                |
| is_initialized                        | bool      | Flag indicating if the configuration has already been initialized. If true, config initialization will be skipped.                                                 |
| version                               | str       | Version of cesnet-tszoo this config was made in.                                                                                                                   |
| export_update_needed                  | bool      | Whether config was updated to newer version and should be exported.                                                                                                |
