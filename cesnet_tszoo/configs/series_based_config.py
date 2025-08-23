import logging
from typing import Literal
from datetime import datetime
from numbers import Number

import numpy as np
import numpy.typing as npt

from cesnet_tszoo.utils.filler import filler_from_input_to_type
from cesnet_tszoo.utils.transformer import transformer_from_input_to_transformer_type, Transformer
from cesnet_tszoo.utils.utils import get_abbreviated_list_string
from cesnet_tszoo.utils.enums import FillerType, TransformerType, TimeFormat, DataloaderOrder, DatasetType
from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.configs.handlers.series_based_handler import SeriesBasedHandler
from cesnet_tszoo.configs.handlers.time_based_handler import TimeBasedHandler


class SeriesBasedConfig(SeriesBasedHandler, DatasetConfig):
    """
    This class is used for configuring the [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset].

    Used to configure the following:

    - Train, validation, test, all sets (time period, sizes, features)
    - Handling missing values (default values, [`fillers`][cesnet_tszoo.utils.filler])
    - Data transformation using [`transformers`][cesnet_tszoo.utils.transformer]
    - Dataloader options (train/val/test/all/init workers, batch size, train loading order)
    - Plotting

    **Important Notes:**

    - Custom fillers must inherit from the [`fillers`][cesnet_tszoo.utils.filler.Filler] base class.
    - It is recommended to use the [`transformers`][cesnet_tszoo.utils.transformer.Transformer] base class, though this is not mandatory as long as it meets the required methods.
        - If a transformer is already initialized and `partial_fit_initialized_transformers` is `False`, the transformer does not require `partial_fit`.
        - Otherwise, the transformer must support `partial_fit`.
        - Transformers must implement `transform` method.
        - Both `partial_fit` and `transform` methods must accept an input of type `np.ndarray` with shape `(times, features)`.
    - `train_ts`, `val_ts`, and `test_ts` must not contain any overlapping time series IDs.

    For available configuration options, refer to [here][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig--configuration-options].

    Attributes:
        used_train_workers: Tracks the number of train workers in use. Helps determine if the train dataloader should be recreated based on worker changes.
        used_val_workers: Tracks the number of validation workers in use. Helps determine if the validation dataloader should be recreated based on worker changes.
        used_test_workers: Tracks the number of test workers in use. Helps determine if the test dataloader should be recreated based on worker changes.
        used_all_workers: Tracks the total number of all workers in use. Helps determine if the all dataloader should be recreated based on worker changes.
        uses_all_ts: Whether all time series set should be used.
        import_identifier: Tracks the name of the config upon import. None if not imported.
        logger: Logger for displaying information.   

    The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize] is called:

    Attributes:
        all_ts: If no specific sets (train/val/test) are provided, all time series IDs are used. When any set is defined, only the time series IDs in defined sets are used.
        train_ts_row_ranges: Initialized when `train_ts` is set. Contains time series IDs in train set with their respective time ID ranges.
        val_ts_row_ranges: Initialized when `val_ts` is set. Contains time series IDs in validation set with their respective time ID ranges.
        test_ts_row_ranges: Initialized when `test_ts` is set. Contains time series IDs in test set with their respective time ID ranges.
        all_ts_row_ranges: Initialized when `all_ts` is set. Contains time series IDs in all set with their respective time ID ranges.
        display_time_period: Used to display the configured value of `time_period`.

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
        train_fillers: Fillers used in the train set. `None` if no filler is used or train set is not used.
        val_fillers: Fillers used in the validation set. `None` if no filler is used or validation set is not used.
        test_fillers: Fillers used in the test set. `None` if no filler is used or test set is not used.
        all_fillers: Fillers used for the all set. `None` if no filler is used or all set is not used.
        is_initialized: Flag indicating if the configuration has already been initialized. If true, config initialization will be skipped.  
        version: Version of cesnet-tszoo this config was made in.
        export_update_needed: Whether config was updated to newer version and should be exported.      

    # Configuration options

    Attributes:
        time_period: Defines the time period for returning data from `train/val/test/all`. Can be a range of time IDs, a tuple of datetime objects or a float. Float value is equivalent to percentage of available times from start.
        train_ts: Defines which time series IDs are used in the training set. Can be a list of IDs, or an integer/float to specify a random selection. An `int` specifies the number of random time series, and a `float` specifies the proportion of available time series. 
                  `int` and `float` must be greater than 0, and a float should be smaller or equal to 1.0. Using `int` or `float` guarantees that no time series from other sets will be used. `Default: None`
        val_ts: Defines which time series IDs are used in the validation set. Same as `train_ts` but for the validation set. `Default: None`
        test_ts: Defines which time series IDs are used in the test set. Same as `train_ts` but for the test set. `Default: None`           
        features_to_take: Defines which features are used. `Default: "all"`
        default_values: Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature. `Default: "default"`
        train_batch_size: Batch size for the train dataloader. Affects number of returned time series in one batch. `Default: 32`
        val_batch_size: Batch size for the validation dataloader. Affects number of returned time series in one batch. `Default: 64`
        test_batch_size: Batch size for the test dataloader. Affects number of returned time series in one batch. `Default: 128`
        all_batch_size: Batch size for the all dataloader. Affects number of returned time series in one batch. `Default: 128`         
        fill_missing_with: Defines how to fill missing values in the dataset. Can pass enum [`FillerType`][cesnet_tszoo.utils.enums.FillerType] for built-in filler or pass a type of custom filler that must derive from [`Filler`][cesnet_tszoo.utils.filler.Filler] base class. `Default: None`
        transform_with: Defines the transformer used to transform the dataset. Can pass enum [`TransformerType`][cesnet_tszoo.utils.enums.TransformerType] for built-in transformer, pass a type of custom transformer or instance of already fitted transformer. `Default: None`
        partial_fit_initialized_transformer: If `True`, partial fitting on train set is performed when using initiliazed transformer. `Default: False`
        include_time: If `True`, time data is included in the returned values. `Default: True`
        include_ts_id: If `True`, time series IDs are included in the returned values. `Default: True`
        time_format: Format for the returned time data. When using TimeFormat.DATETIME, time will be returned as separate list along rest of the values. `Default: TimeFormat.ID_TIME`
        train_workers: Number of workers for loading training data. `0` means that the data will be loaded in the main process. `Default: 4`
        val_workers: Number of workers for loading validation data. `0` means that the data will be loaded in the main process. `Default: 3`
        test_workers: Number of workers for loading test data. `0` means that the data will be loaded in the main process. `Default: 2`
        all_workers: Number of workers for loading all data. `0` means that the data will be loaded in the main process. `Default: 4`
        init_workers: Number of workers for initial dataset processing during configuration. `0` means that the data will be loaded in the main process. `Default: 4`
        nan_threshold: Maximum allowable percentage of missing data. Time series exceeding this threshold are excluded. Time series over the threshold will not be used. Used for `train/val/test/all` separately. `Default: 1.0`
        train_dataloader_order: Defines the order of data returned by the training dataloader. `Default: DataloaderOrder.SEQUENTIAL`
        random_state: Fixes randomness for reproducibility during configuration and dataset initialization. `Default: None`                  
    """

    def __init__(self,
                 time_period: tuple[datetime, datetime] | range | float | Literal["all"],
                 train_ts: list[int] | npt.NDArray[np.int_] | float | int | None = None,
                 val_ts: list[int] | npt.NDArray[np.int_] | float | int | None = None,
                 test_ts: list[int] | npt.NDArray[np.int_] | float | int | None = None,
                 features_to_take: list[str] | Literal["all"] = "all",
                 default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None = "default",
                 train_batch_size: int = 32,
                 val_batch_size: int = 64,
                 test_batch_size: int = 128,
                 all_batch_size: int = 128,
                 fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None = None,
                 transform_with: type | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "l2_normalizer"] | None = None,
                 partial_fit_initialized_transformer: bool = False,
                 include_time: bool = True,
                 include_ts_id: bool = True,
                 time_format: TimeFormat | Literal["id_time", "datetime", "unix_time", "shifted_unix_time"] = TimeFormat.ID_TIME,
                 train_workers: int = 4,
                 val_workers: int = 3,
                 test_workers: int = 2,
                 all_workers: int = 4,
                 init_workers: int = 4,
                 nan_threshold: float = 1.0,
                 train_dataloader_order: DataloaderOrder | Literal["random", "sequential"] = DataloaderOrder.SEQUENTIAL,
                 random_state: int | None = None):

        self.time_period = time_period

        self.display_time_period = None

        self.logger = logging.getLogger("series_config")

        SeriesBasedHandler.__init__(self, self.logger, True, train_ts, val_ts, test_ts)
        DatasetConfig.__init__(self, features_to_take, default_values, train_batch_size, val_batch_size, test_batch_size, all_batch_size, fill_missing_with, transform_with, partial_fit_initialized_transformer, include_time, include_ts_id, time_format,
                               train_workers, val_workers, test_workers, all_workers, init_workers, nan_threshold, False, DatasetType.SERIES_BASED, train_dataloader_order, random_state, self.logger)

    def _validate_construction(self) -> None:
        """Performs basic parameter validation to ensure correct configuration. More comprehensive validation, which requires dataset-specific data, is handled in [`_dataset_init`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig._dataset_init]. """

        DatasetConfig._validate_construction(self)

        if isinstance(self.time_period, (float, int)):
            self.time_period = float(self.time_period)
            assert self.time_period > 0.0, "time_period must be greater than 0"
            assert self.time_period <= 1.0, "time_period must be lower or equal to 1.0"

        self._validate_ts_init()

        self.logger.debug("Series-based configuration validated successfully.")

    def _get_train(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the training set. """
        return self.train_ts, self.time_period

    def _get_val(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the validation set. """
        return self.val_ts, self.time_period

    def _get_test(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the test set. """
        return self.test_ts, self.time_period

    def _get_all(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the all set. """
        return self.all_ts, self.time_period

    def has_train(self) -> bool:
        """Returns whether training set is used. """
        return self.train_ts is not None

    def has_val(self) -> bool:
        """Returns whether validation set is used. """
        return self.val_ts is not None

    def has_test(self) -> bool:
        """Returns whether test set is used. """
        return self.test_ts is not None

    def has_all(self) -> bool:
        """Returns whether all set is used. """
        return self.all_ts is not None

    def _set_time_period(self, all_time_ids: np.ndarray) -> None:
        """Validates and filters the input time period based on the dataset and aggregation. """

        if self.time_period == "all":
            self.time_period = range(len(all_time_ids))
            self.logger.debug("Time period set to 'all'. Using all available time IDs, range: %s", self.time_period)
        elif isinstance(self.time_period, float):
            self.time_period = range(int(self.time_period * len(all_time_ids)))
            self.logger.debug("Time period set with float value. Using range: %s", self.time_period)

        self.time_period, self.display_time_period = TimeBasedHandler._process_time_period(self.time_period, all_time_ids, self.logger, self.time_format)
        self.logger.debug("Processed time_period: %s, display_time_period: %s", self.time_period, self.display_time_period)

    def _set_ts(self, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray) -> None:
        """Validates and filters the input time series IDs based on the `dataset` and `source_type`. Handles random split."""

        self._prepare_and_set_ts_sets(all_ts_ids, all_ts_row_ranges, self.ts_id_name, self.random_state)

    def _set_feature_transformers(self) -> None:
        """Creates and/or validates transformers based on the `transform_with` parameter. """

        self.create_transformer_per_time_series = False

        if self.transform_with is None:
            self.transform_with_display = None
            self.are_transformers_premade = False
            self.transformers = None
            self.is_transformer_custom = None

            self.logger.debug("No transformer will be used because transform_with is not set.")
            return

        # Treat transform_with as already initialized transformer
        if not isinstance(self.transform_with, (type, TransformerType)):
            self.transformers = self.transform_with

            if not self.has_train():
                if self.partial_fit_initialized_transformers:
                    self.logger.warning("partial_fit_initialized_transformers will be ignored because train set is not used.")
                self.partial_fit_initialized_transformers = False

            self.transform_with, self.transform_with_display = transformer_from_input_to_transformer_type(type(self.transform_with), check_for_fit=False, check_for_partial_fit=self.partial_fit_initialized_transformers)

            self.are_transformers_premade = True

            self.is_transformer_custom = "Custom" in self.transform_with_display
            self.logger.debug("Using initialized transformer of type: %s", self.transform_with_display)

        # Treat transform_with as uninitialized transformer
        else:
            if not self.has_train():
                self.transform_with = None
                self.transform_with_display = None
                self.are_transformers_premade = False
                self.transformers = None
                self.is_transformer_custom = None

                self.logger.warning("No transformer will be used because train set is not used.")
                return

            self.transform_with, self.transform_with_display = transformer_from_input_to_transformer_type(self.transform_with, check_for_fit=False, check_for_partial_fit=True)
            self.transformers = self.transform_with()

            self.are_transformers_premade = False

            self.is_transformer_custom = "Custom" in self.transform_with_display
            self.logger.debug("Using uninitialized transformer of type: %s", self.transform_with_display)

    def _set_fillers(self) -> None:
        """Creates and/or validates fillers based on the `fill_missing_with` parameter. """

        self.fill_missing_with, self.fill_missing_with_display = filler_from_input_to_type(self.fill_missing_with)
        self.is_filler_custom = "Custom" in self.fill_missing_with_display if self.fill_missing_with is not None else None

        if self.fill_missing_with is None:
            self.logger.debug("No filler is used because fill_missing_with is set to None.")
            return

        # Set the fillers for the training set
        if self.has_train():
            self.train_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.train_ts])
            self.logger.debug("Fillers for training set are set.")

        # Set the fillers for the validation set
        if self.has_val():
            self.val_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.val_ts])
            self.logger.debug("Fillers for validation set are set.")

        # Set the fillers for the test set
        if self.has_test():
            self.test_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.test_ts])
            self.logger.debug("Fillers for test set are set.")

        # Set the fillers for the all set
        if self.has_all():
            self.all_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.all_ts])
            self.logger.debug("Fillers for all set are set.")

        self.logger.debug("Using filler: %s.", self.fill_missing_with_display)

    def _validate_finalization(self) -> None:
        """Performs final validation of the configuration. """

        self._validate_ts_overlap()

    def __str__(self) -> str:

        if self.transform_with is None:
            transformer_part = f"Transformer type: {str(self.transform_with_display)}"
        else:
            transformer_part = f'''Transformer type: {str(self.transform_with_display)}
        Are transformers premade: {self.are_transformers_premade}
        Are premade transformers partial_fitted: {self.partial_fit_initialized_transformers}'''

        if self.include_time:
            time_part = f'''Time included: {str(self.include_time)}    
        Time format: {str(self.time_format)}'''
        else:
            time_part = f"Time included: {str(self.include_time)}"

        return f'''
Config Details:
    Used for database: {self.database_name}
    Aggregation: {str(self.aggregation)}
    Source: {str(self.source_type)}

    Time series
        Train time series IDS: {get_abbreviated_list_string(self.train_ts)}
        Val time series IDS: {get_abbreviated_list_string(self.val_ts)}
        Test time series IDS {get_abbreviated_list_string(self.test_ts)}
        All time series IDS {get_abbreviated_list_string(self.all_ts)}
    Time periods
        Time period: {str(self.display_time_period)}
    Features
        Taken features: {str(self.features_to_take_without_ids)}
        Default values: {self.default_values}
        Time series ID included: {str(self.include_ts_id)}
        {time_part}
    Fillers         
        Filler type: {str(self.fill_missing_with_display)}
    Transformers
        {transformer_part}
    Batch sizes
        Train batch size: {self.train_batch_size}
        Val batch size: {self.val_batch_size}
        Test batch size: {self.test_batch_size}
        All batch size: {self.all_batch_size}
    Default workers
        Train worker count: {str(self.train_workers)}
        Val worker count: {str(self.val_workers)}
        Test worker count: {str(self.test_workers)}
        All worker count: {str(self.all_workers)}
        Init worker count: {str(self.init_workers)}
    Other
        Nan threshold: {str(self.nan_threshold)}
        Random state: {self.random_state}
        Train dataloader order: {str(self.train_dataloader_order)}
        Version: {self.version}
                '''
