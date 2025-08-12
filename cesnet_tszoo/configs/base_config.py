from typing import Literal
from numbers import Number
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import math
import logging

import numpy as np
import numpy.typing as npt
from packaging.version import Version
from sklearn.model_selection import train_test_split

import cesnet_tszoo.version as version
from cesnet_tszoo.utils.constants import ROW_END, ROW_START, ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
from cesnet_tszoo.utils.enums import AgreggationType, FillerType, TimeFormat, TransformerType, DataloaderOrder, ScalerType, DatasetType
from cesnet_tszoo.utils.transformer import Transformer


class DatasetConfig(ABC):
    """
    Base class for configuration management. This class should **not** be used directly. Instead, use one of its derived classes, such as [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig] or [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig].

    For available configuration options, refer to [here][cesnet_tszoo.configs.base_config.DatasetConfig--configuration-options].

    Attributes:
        used_train_workers: Tracks the number of train workers in use. Helps determine if the train dataloader should be recreated based on worker changes.
        used_val_workers: Tracks the number of validation workers in use. Helps determine if the validation dataloader should be recreated based on worker changes.
        used_test_workers: Tracks the number of test workers in use. Helps determine if the test dataloader should be recreated based on worker changes.
        used_all_workers: Tracks the total number of all workers in use. Helps determine if the all dataloader should be recreated based on worker changes.
        import_identifier: Tracks the name of the config upon import. None if not imported.
        logger: Logger for displaying information. 

    The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize] is called:

    Attributes:
        aggregation: The aggregation period used for the data.
        source_type: The source type of the data.
        database_name: Specifies which database this config applies to.
        transform_with_display: Used to display the configured type of `transform_with`.
        fill_missing_with_display: Used to display the configured type of `fill_missing_with`.
        features_to_take_without_ids: Features to be returned, excluding time or time series IDs.
        indices_of_features_to_take_no_ids: Indices of non-ID features in `features_to_take`.
        is_transformer_custom: Flag indicating whether the transformer is custom.
        is_filler_custom: Flag indicating whether the filler is custom.
        ts_id_name: Name of the time series ID, dependent on `source_type`.
        used_times: List of all times used in the configuration.
        used_ts_ids: List of all time series IDs used in the configuration.
        used_ts_row_ranges: List of time series IDs with their respective time ID ranges.
        used_fillers: List of all fillers used in the configuration.
        used_singular_train_time_series: Currently used singular train set time series for dataloader.
        used_singular_val_time_series: Currently used singular validation set time series for dataloader.
        used_singular_test_time_series: Currently used singular test set time series for dataloader.
        used_singular_all_time_series: Currently used singular all set time series for dataloader.        
        transformers: Prepared transformers for fitting/transforming. Can be one transformer, array of transformers or `None`.
        are_transformers_premade: Indicates whether the transformers are premade.
        has_train: Flag indicating whether the training set is in use.
        has_val: Flag indicating whether the validation set is in use.
        has_test: Flag indicating whether the test set is in use.
        has_all: Flag indicating whether the all set is in use.
        train_fillers: Fillers used in the train set. `None` if no filler is used or train set is not used.
        val_fillers: Fillers used in the validation set. `None` if no filler is used or validation set is not used.
        test_fillers: Fillers used in the test set. `None` if no filler is used or test set is not used.
        all_fillers: Fillers used for the all set. `None` if no filler is used or all set is not used.
        is_initialized: Flag indicating if the configuration has already been initialized. If true, config initialization will be skipped.  
        version: Version of cesnet-tszoo this config was made in.
        export_update_needed: Whether config was updated to newer version and should be exported.

    # Configuration options

    Attributes:
        features_to_take: Defines which features are used.
        default_values: Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature.
        sliding_window_size: Number of values in one window. Impacts dataloader behavior.
        sliding_window_prediction_size: Number of times to predict from sliding_window_size. Impacts dataloader behavior.
        sliding_window_step: Number of times to move by after each window.
        set_shared_size: How much times should time periods share.
        train_batch_size: Batch size for the train dataloader, when window size is None.
        val_batch_size: Batch size for the validation dataloader, when window size is None.
        test_batch_size: Batch size for the test dataloader, when window size is None.
        all_batch_size: Batch size for the all dataloader, when window size is None.
        fill_missing_with: Defines how to fill missing values in the dataset. Can pass enum [`FillerType`][cesnet_tszoo.utils.enums.FillerType] for built-in filler or pass a type of custom filler that must derive from [`Filler`][cesnet_tszoo.utils.filler.Filler] base class.
        transform_with: Defines the transformer to transform the dataset. Can pass enum [`TransformerType`][cesnet_tszoo.utils.enums.TransformerType] for built-in transformer, pass a type of custom transformer or instance of already fitted transformer(s).
        partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers.
        include_time: If `True`, time data is included in the returned values.
        include_ts_id: If `True`, time series IDs are included in the returned values.
        time_format: Format for the returned time data. When using TimeFormat.DATETIME, time will be returned as separate list along rest of the values.
        train_workers: Number of workers for loading training data. `0` means that the data will be loaded in the main process.
        val_workers: Number of workers for loading validation data. `0` means that the data will be loaded in the main process.
        test_workers: Number of workers for loading test data. `0` means that the data will be loaded in the main process.
        all_workers: Number of workers for loading all data. `0` means that the data will be loaded in the main process.
        init_workers: Number of workers for initial dataset processing during configuration. `0` means that the data will be loaded in the main process.
        nan_threshold: Maximum allowable percentage of missing data. Time series exceeding this threshold are excluded. Time series over the threshold will not be used. Used for `train/val/test/all` separately.
        create_transformer_per_time_series: If `True`, a separate transformer is created for each time series. Not used when using already initialized transformers. 
        is_series_based: Flag indicating if the config applies to a series-based dataset.
        train_dataloader_order: Defines the order of data returned by the training dataloader.
        random_state: Fixes randomness for reproducibility during configuration and dataset initialization.              
    """

    def __init__(self,
                 features_to_take: list[str] | Literal["all"],
                 default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 all_batch_size: int,
                 fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None,
                 transform_with: type | TransformerType | list[Transformer] | np.ndarray[Transformer] | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None,
                 partial_fit_initialized_transformers: bool,
                 include_time: bool,
                 include_ts_id: bool,
                 time_format: TimeFormat | Literal["id_time", "datetime", "unix_time", "shifted_unix_time"],
                 train_workers: int,
                 val_workers: int,
                 test_workers: int,
                 all_workers: int,
                 init_workers: int,
                 nan_threshold: float,
                 create_transformer_per_time_series: bool,
                 dataset_type: DatasetType,
                 train_dataloader_order: DataloaderOrder | Literal["random", "sequential"],
                 random_state: int | None,
                 logger: logging.Logger):

        self.features_to_take = features_to_take
        self.default_values = default_values
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.all_batch_size = all_batch_size
        self.fill_missing_with = fill_missing_with
        self.transform_with = transform_with
        self.partial_fit_initialized_transformers = partial_fit_initialized_transformers
        self.include_time = include_time
        self.include_ts_id = include_ts_id
        self.time_format = time_format
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers
        self.all_workers = all_workers
        self.init_workers = init_workers
        self.nan_threshold = nan_threshold
        self.create_transformer_per_time_series = create_transformer_per_time_series
        self.dataset_type = dataset_type
        self.train_dataloader_order = train_dataloader_order
        self.random_state = random_state

        self.used_train_workers = None
        self.used_val_workers = None
        self.used_test_workers = None
        self.used_all_workers = None
        self.import_identifier = None

        self.aggregation = None
        self.source_type = None
        self.database_name = None
        self.transform_with_display = None
        self.fill_missing_with_display = None
        self.features_to_take_without_ids = None
        self.indices_of_features_to_take_no_ids = None
        self.is_transformer_custom = False
        self.is_filler_custom = False
        self.ts_id_name = None
        self.used_times = None
        self.used_ts_ids = None
        self.used_ts_row_ranges = None
        self.used_fillers = None
        self.used_singular_train_time_series = None
        self.used_singular_val_time_series = None
        self.used_singular_test_time_series = None
        self.used_singular_all_time_series = None
        self.transformers = None
        self.are_transformers_premade = False
        self.train_fillers = None
        self.val_fillers = None
        self.test_fillers = None
        self.all_fillers = None
        self.is_initialized = False
        self.version = version.current_version
        self.export_update_needed = False
        self.logger = logger

        self._validate_construction()

        self.logger.info("Quick validation succeeded.")

    def _validate_construction(self) -> None:
        """Performs basic parameter validation to ensure correct configuration. More comprehensive validation, which requires dataset-specific data, is handled in [`_dataset_init`][cesnet_tszoo.configs.base_config.DatasetConfig._dataset_init]. """

        # Ensuring boolean flags are correctly set
        assert isinstance(self.partial_fit_initialized_transformers, bool), "partial_fit_initialized_transformers must be a boolean value."
        assert isinstance(self.include_time, bool), "include_time must be a boolean value."
        assert isinstance(self.include_ts_id, bool), "include_ts_id must be a boolean value."
        assert isinstance(self.create_transformer_per_time_series, bool), "create_transformer_per_time_series must be a boolean value."

        # Ensuring worker count values are non-negative integers
        assert isinstance(self.train_workers, int) and self.train_workers >= 0, "train_workers must be a non-negative integer."
        assert isinstance(self.val_workers, int) and self.val_workers >= 0, "val_workers must be a non-negative integer."
        assert isinstance(self.test_workers, int) and self.test_workers >= 0, "test_workers must be a non-negative integer."
        assert isinstance(self.all_workers, int) and self.all_workers >= 0, "all_workers must be a non-negative integer."
        assert isinstance(self.init_workers, int) and self.init_workers >= 0, "init_workers must be a non-negative integer."

        # Ensuring batch size values are positive integers
        assert isinstance(self.train_batch_size, int) and self.train_batch_size > 0, "train_batch_size must be a positive integer."
        assert isinstance(self.val_batch_size, int) and self.val_batch_size > 0, "val_batch_size must be a positive integer."
        assert isinstance(self.test_batch_size, int) and self.test_batch_size > 0, "test_batch_size must be a positive integer."
        assert isinstance(self.all_batch_size, int) and self.all_batch_size > 0, "all_batch_size must be a positive integer."

        if isinstance(self.set_shared_size, float):
            assert self.set_shared_size >= 0 and self.set_shared_size <= 1, "set_shared_size float value must be between or equal to 0 and 1."

        assert self.set_shared_size >= 0, "set_shared_size must be of positive value."

        # Ensure sliding_window_size is either None or a valid integer greater than 1
        assert self.sliding_window_size is None or (isinstance(self.sliding_window_size, int) and self.sliding_window_size > 1), "sliding_window_size must be an integer greater than 1, or None."

        # Ensure sliding_window_prediction_size is either None or a valid integer greater or equal to 1
        assert self.sliding_window_prediction_size is None or (isinstance(self.sliding_window_prediction_size, int) and self.sliding_window_prediction_size >= 1), "sliding_window_prediction_size must be an integer greater than 1, or None."

        # Both sliding_window_size and sliding_window_prediction_size must be set or None
        assert (self.sliding_window_size is None and self.sliding_window_prediction_size is None) or (self.sliding_window_size is not None and self.sliding_window_prediction_size is not None), "Both sliding_window_size and sliding_window_prediction_size must be set or None."

        # Adjust batch sizes based on sliding_window_size
        if self.sliding_window_size is not None:

            if self.sliding_window_step <= 0:
                raise ValueError("sliding_window_step must be greater or equal to 1.")

            total_window_size = self.sliding_window_size + self.sliding_window_prediction_size

            if total_window_size > self.train_batch_size:
                self.train_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("train_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > self.val_batch_size:
                self.val_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("val_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > self.test_batch_size:
                self.test_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("test_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > self.all_batch_size:
                self.all_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("all_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)

        # Validate nan_threshold value
        assert isinstance(self.nan_threshold, Number) and 0 <= self.nan_threshold <= 1, "nan_threshold must be a number between 0 and 1."
        self.nan_threshold = float(self.nan_threshold)

        # Convert time_format and train_dataloader_order to their respective enum types
        self.time_format = TimeFormat(self.time_format)
        self.train_dataloader_order = DataloaderOrder(self.train_dataloader_order)

        # Validate and process transformer type
        if isinstance(self.transform_with, (str, TransformerType)):
            self.transform_with = TransformerType(self.transform_with)
            if self.transform_with in [TransformerType.POWER_TRANSFORMER, TransformerType.QUANTILE_TRANSFORMER, TransformerType.ROBUST_SCALER] and not self.create_transformer_per_time_series:
                raise NotImplementedError("The selected transformer requires a working partial_fit method, which is not implemented for this configuration.")

        # Validate and process missing data filler type
        if isinstance(self.fill_missing_with, (str, FillerType)):
            self.fill_missing_with = FillerType(self.fill_missing_with)

    def _update_sliding_window(self, sliding_window_size: int | None, sliding_window_prediction_size: int | None, sliding_window_step: int | None, set_shared_size: float | int, all_time_ids: np.ndarray):
        return

    def _update_batch_sizes(self, train_batch_size: int, val_batch_size: int, test_batch_size: int, all_batch_size: int) -> None:

        # Ensuring batch size values are positive integers
        assert isinstance(train_batch_size, int) and train_batch_size > 0, "train_batch_size must be a positive integer."
        assert isinstance(val_batch_size, int) and val_batch_size > 0, "val_batch_size must be a positive integer."
        assert isinstance(test_batch_size, int) and test_batch_size > 0, "test_batch_size must be a positive integer."
        assert isinstance(all_batch_size, int) and all_batch_size > 0, "all_batch_size must be a positive integer."

        # Adjust batch sizes based on sliding_window_size
        if self.sliding_window_size is not None:

            if self.sliding_window_step <= 0:
                raise ValueError("sliding_window_step must be greater or equal to 1.")

            total_window_size = self.sliding_window_size + self.sliding_window_prediction_size

            if total_window_size > train_batch_size:
                train_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("train_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > val_batch_size:
                val_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("val_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > test_batch_size:
                test_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("test_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > all_batch_size:
                all_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("all_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.all_batch_size = all_batch_size

        self.logger.debug("Updated batch sizes.")

    def _update_workers(self, train_workers: int, val_workers: int, test_workers: int, all_workers: int, init_workers: int) -> None:

        # Ensuring worker count values are non-negative integers
        assert isinstance(self.train_workers, int) and self.train_workers >= 0, "train_workers must be a non-negative integer."
        assert isinstance(self.val_workers, int) and self.val_workers >= 0, "val_workers must be a non-negative integer."
        assert isinstance(self.test_workers, int) and self.test_workers >= 0, "test_workers must be a non-negative integer."
        assert isinstance(self.all_workers, int) and self.all_workers >= 0, "all_workers must be a non-negative integer."
        assert isinstance(self.init_workers, int) and self.init_workers >= 0, "init_workers must be a non-negative integer."

        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers
        self.all_workers = all_workers
        self.init_workers = init_workers

        self.logger.debug("Updated workers.")

    @abstractmethod
    def _get_train(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the training set. """
        ...

    @abstractmethod
    def _get_val(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the validation set. """
        ...

    @abstractmethod
    def _get_test(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the test set. """
        ...

    @abstractmethod
    def _get_all(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the all set. """
        ...

    @abstractmethod
    def has_train(self) -> bool:
        """Returns whether training set is used. """
        ...

    @abstractmethod
    def has_val(self) -> bool:
        """Returns whether validation set is used. """
        ...

    @abstractmethod
    def has_test(self) -> bool:
        """Returns whether test set is used. """
        ...

    @abstractmethod
    def has_all(self) -> bool:
        """Returns whether all set is used. """
        ...

    def _get_table_data_path(self) -> str:
        """Returns the path to the data table corresponding to the `source_type` and `aggregation`."""
        return f"/{self.source_type.value}/{AgreggationType._to_str_with_agg(self.aggregation)}"

    def _get_table_identifiers_row_ranges_path(self) -> str:
        """Returns the path to the identifiers' row ranges table corresponding to the `source_type` and `aggregation`. """
        return f"/{self.source_type.value}/id_ranges_{AgreggationType._to_str_with_agg(self.aggregation)}"

    def _dataset_init(self, all_real_ts_ids: np.ndarray, all_time_ids: np.ndarray, all_ts_row_ranges: np.ndarray, all_dataset_features: dict[str, np.dtype], default_values: dict[str, Number], ts_id_name: str) -> None:
        """Performs deeper parameter validation and updates values based on data from the dataset. """

        self.ts_id_name = ts_id_name

        # Set the features to take
        self._set_features_to_take(all_dataset_features)
        self.logger.debug("Features to take have been successfully set.")

        # Set time series IDs
        self._set_ts(all_real_ts_ids, all_ts_row_ranges)
        self.logger.debug("Time series IDs have been successfully set.")

        # Set the time periods
        self._set_time_period(all_time_ids)
        self.logger.debug("Time period have been successfully set.")

        # Set default values
        self._set_default_values(default_values)
        self.logger.debug("Default values have been successfully set.")

        # Set feature transformers
        self._set_feature_transformers()
        self.logger.debug("Feature transformers have been successfully set.")

        # Set fillers
        self._set_fillers()
        self.logger.debug("Fillers have been successfully set.")

        # Final validation and finalization
        self._validate_finalization()

        self.logger.info("Finalization and validation completed successfully.")

    def _set_features_to_take(self, all_dataset_features: dict[str, np.dtype]) -> None:
        """Validates and filters the input `features_to_take` based on the `dataset`, `source_type`, and `aggregation`. """

        if self.features_to_take == "all":
            self.features_to_take = list(all_dataset_features.keys())
            self.logger.debug("All features used because 'features_to_take' is set to 'all'.")

        # Handling the inclusion of time ID in features
        if self.include_time and self.features_to_take.count(ID_TIME_COLUMN_NAME) == 0 and self.time_format != TimeFormat.DATETIME:
            self.features_to_take.insert(0, ID_TIME_COLUMN_NAME)
            self.logger.debug("Added '%s' to the features as 'include_time' is true and 'time_format' is not datetime.", ID_TIME_COLUMN_NAME)
        elif self.include_time and self.features_to_take.count(ID_TIME_COLUMN_NAME) > 0 and self.time_format == TimeFormat.DATETIME:
            self.features_to_take.remove(ID_TIME_COLUMN_NAME)
            self.logger.debug("Removed '%s' from the features because 'time_format' is datetime.", ID_TIME_COLUMN_NAME)
        elif not self.include_time and self.features_to_take.count(ID_TIME_COLUMN_NAME) > 0:
            self.features_to_take.remove(ID_TIME_COLUMN_NAME)
            self.logger.debug("Removed '%s' from the features as 'include_time' is false.", ID_TIME_COLUMN_NAME)

        # Handling the inclusion of time series ID feature
        if self.include_ts_id and self.features_to_take.count(self.ts_id_name) <= 0:
            self.features_to_take.insert(0, self.ts_id_name)
            self.logger.debug("Added '%s' to the features as 'include_ts_id' is true.", self.ts_id_name)
        elif not self.include_ts_id and self.features_to_take.count(self.ts_id_name) > 0:
            self.features_to_take.remove(self.ts_id_name)
            self.logger.debug("Removed '%s' from the features as 'include_ts_id' is false.", self.ts_id_name)

        # Filtering features based on available dataset features
        temp = list(self.features_to_take)
        self.features_to_take = [feature for feature in self.features_to_take if feature in all_dataset_features or feature == ID_TIME_COLUMN_NAME or feature == self.ts_id_name]

        if len(temp) != len(self.features_to_take):
            self.logger.warning("Some features were removed as they are not available in the dataset.")

        # Preparing indices and features without time and time series ID
        self.indices_of_features_to_take_no_ids = [idx for idx, feature in enumerate(self.features_to_take) if feature != ID_TIME_COLUMN_NAME and feature != self.ts_id_name]
        self.features_to_take_without_ids = [feature for feature in self.features_to_take if feature != ID_TIME_COLUMN_NAME and feature != self.ts_id_name]

        # Assert that at least one feature is used
        assert len(self.features_to_take_without_ids) > 0, "At least one non-ID feature must be used."

    def _set_default_values(self, default_values: dict[str, Number]) -> None:
        """Validates and filters the input `default_values` based on the `dataset`, `source_type`, `aggregation`, and `features_to_take`. """

        if self.default_values == "default":
            self.default_values = dict(default_values)
            self.logger.debug("Using default dataset values for default values because 'default_values' is set to 'default'.")

        elif isinstance(self.default_values, Number):
            # If default_values is a single number, assign it to all features

            orig_default_value = self.default_values
            self.default_values = {feature: float(self.default_values) for feature in self.features_to_take_without_ids}
            self.logger.debug("Assigned the default value %s to all features as 'default_values' is a single number.", float(orig_default_value))

        elif isinstance(self.default_values, (list, np.ndarray)):
            # If default_values is a list or ndarray, ensure the length matches with features_to_take_without_ids
            if len(self.default_values) != len(self.features_to_take_without_ids):
                raise ValueError("The number of values in 'default_values' does not match the number of features in 'features_to_take'.")
            self.default_values = {feature: value for feature, value in zip(self.features_to_take_without_ids, self.default_values) if feature != ID_TIME_COLUMN_NAME and feature != self.ts_id_name}
            self.logger.debug("Mapped default values to features, skipping IDs features: %s", self.default_values)

        elif isinstance(self.default_values, dict):
            # If default_values is a dictionary, ensure its keys match the features
            if set(self.default_values.keys()) != set(self.features_to_take_without_ids):
                raise ValueError("The keys in 'default_values' do not match the features in 'features_to_take'.")
            self.logger.debug("Using provided default values for features: %s", self.default_values)

        elif self.default_values is None or math.isnan(self.default_values) or np.isnan(self.default_values):
            # If default_values is None or NaN, assign NaN to each feature
            self.default_values = {feature: np.nan for feature in self.features_to_take_without_ids}
            self.logger.debug("Assigned NaN as the default value for all features because 'default_values' is None or NaN.")

        # Convert the default values into a NumPy array for consistent data handling
        temp_default_values = np.ndarray(len(self.features_to_take_without_ids), np.float64)
        for i, feature in enumerate(self.features_to_take_without_ids):
            temp_default_values[i] = self.default_values[feature]

        self.default_values = temp_default_values

    @abstractmethod
    def _set_time_period(self, all_time_ids: np.ndarray) -> None:
        """Validates and filters the input time periods based on the dataset and aggregation. This typically calls [`_process_time_period`][cesnet_tszoo.configs.base_config.DatasetConfig._process_time_period] for each time period. """
        ...

    @abstractmethod
    def _set_ts(self, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray) -> None:
        """Validates and filters the input time series IDs based on the `dataset` and `source_type`. This typically calls [`_process_ts_ids`][cesnet_tszoo.configs.base_config.DatasetConfig._process_ts_ids] for each time series ID filter. """
        ...

    @abstractmethod
    def _set_feature_transformers(self) -> None:
        """Creates and/or validates transformers based on the `transform_with` parameter. """
        ...

    @abstractmethod
    def _set_fillers(self) -> None:
        """Creates and/or validates fillers based on the `fill_missing_with` parameter. """
        ...

    @abstractmethod
    def _validate_finalization(self) -> None:
        """Performs final validation of the configuration. """
        ...

    def _try_backward_support_update(self) -> None:
        """Tries to update config to match newer version of library. """

        self.logger.debug("Trying to update config if necessary.")

        self._try_set_default_version()

        if Version(self.version) < Version(version.VERSION_0_1_3):
            self.logger.warning("Imported config version is lower than '%s', updating attributes from scaler variant to transformer variant.", version.VERSION_0_1_3)
            self._scaler_to_transformer_version_update()

        self._try_version_update()

    def _try_version_update(self):
        if Version(self.version) < Version(version.current_version):
            self.logger.warning("Imported config was made for cesnet-tszoo package of version '%s', but current used cesnet-tszoo package version is '%s'!", self.version, version.current_version)
            self.logger.warning("Will try to update the config. It is recommended to recreate this config or at least export this config alone or through benchmark to create updated config file.")
            self.export_update_needed = True
        elif Version(self.version) > Version(version.current_version):
            self.logger.error("Imported config was made for cesnet-tszoo package of version '%s', but current used cesnet-tszoo package version is '%s'!", self.version, version.current_version)
            self.logger.error("Update cesnet-tszoo package to use this config.")
            raise ValueError(f"Imported config was made for cesnet-tszoo package of version '{self.version}', but current used cesnet-tszoo package version is '{version.current_version}'!")
        else:
            self.export_update_needed = False

        self.logger.debug("Updating config version to current used cesnet-tszoo package version.")
        self.version = version.current_version

    def _scaler_to_transformer_version_update(self):
        self.transform_with = getattr(self, "scale_with")
        delattr(self, "scale_with")

        if self.transform_with is not None and isinstance(self.transform_with, ScalerType):
            self.transform_with = TransformerType(self.transform_with.value)

        self.partial_fit_initialized_transformers = getattr(self, "partial_fit_initialized_scalers")
        delattr(self, "partial_fit_initialized_scalers")

        self.create_transformer_per_time_series = getattr(self, "create_scaler_per_time_series")
        delattr(self, "create_scaler_per_time_series")

        self.transform_with_display = getattr(self, "scale_with_display")
        delattr(self, "scale_with_display")

        self.is_transformer_custom = getattr(self, "is_scaler_custom")
        delattr(self, "is_scaler_custom")

        self.transformers = getattr(self, "scalers")
        delattr(self, "scalers")

        self.are_transformers_premade = getattr(self, "are_scalers_premade")
        delattr(self, "are_scalers_premade")

    def _try_set_default_version(self):
        if not hasattr(self, "version"):
            self.logger.debug("Config attribute 'version' is missing in this instance. Default version '%s' will be temporarily set.", version.DEFAULT_VERSION)
            self.version = version.DEFAULT_VERSION
