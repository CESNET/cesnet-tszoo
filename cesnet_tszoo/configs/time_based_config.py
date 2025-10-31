import logging
from typing import Literal
from datetime import datetime
from numbers import Number

import numpy as np
import numpy.typing as npt

from cesnet_tszoo.utils.transformer import Transformer
import cesnet_tszoo.utils.transformer.factory as transformer_factories
import cesnet_tszoo.utils.anomaly_handler.factory as anomaly_handler_factories
from cesnet_tszoo.utils.utils import get_abbreviated_list_string
from cesnet_tszoo.utils.enums import FillerType, TransformerType, TimeFormat, DataloaderOrder, DatasetType, AnomalyHandlerType
from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.configs.handlers.series_based_handler import SeriesBasedHandler
from cesnet_tszoo.configs.handlers.time_based_handler import TimeBasedHandler
from cesnet_tszoo.utils.custom_handler.factory import PerSeriesCustomHandlerFactory, NoFitCustomHandlerFactory
from cesnet_tszoo.data_models.holders import PerSeriesCustomHandlerHolder, NoFitCustomHandlerHolder
from cesnet_tszoo.data_models.preprocess_note import PreprocessNote


class TimeBasedConfig(TimeBasedHandler, DatasetConfig):
    """
    This class is used for configuring the [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset].

    Used to configure the following:

    - Train, validation, test, all sets (time period, sizes, features, window size)
    - Handling missing values (default values, [`fillers`][cesnet_tszoo.utils.filler])
    - Handling anomalies ([`anomaly handlers`][cesnet_tszoo.utils.anomaly_handler])
    - Data transformation using [`transformers`][cesnet_tszoo.utils.transformer]
    - Dataloader options (train/val/test/all/init workers, batch sizes)
    - Plotting

    **Important Notes:**

    - Custom fillers must inherit from the [`fillers`][cesnet_tszoo.utils.filler.Filler] base class.
    - Fillers can carry over values from the train set to the validation and test sets. For example, [`ForwardFiller`][cesnet_tszoo.utils.filler.ForwardFiller] can carry over values from previous sets.   
    - Custom anomaly handlers must inherit from the [`anomaly handlers`][cesnet_tszoo.utils.anomaly_handler.AnomalyHandler] base class.
    - It is recommended to use the [`transformers`][cesnet_tszoo.utils.transformer.Transformer] base class, though this is not mandatory as long as it meets the required methods.
        - If transformers are already initialized and `create_transformer_per_time_series` is `True` and `partial_fit_initialized_transformers` is `True` then transformers must support `partial_fit`.
        - If `create_transformer_per_time_series` is `True`, transformers must have a `fit` method and `transform_with` should be a list of transformers.
        - If `create_transformer_per_time_series` is `False`, transformers must support `partial_fit`.
        - Transformers must implement the `transform` method.
        - The `fit/partial_fit` and `transform` methods must accept an input of type `np.ndarray` with shape `(times, features)`.
    - `train_time_period`, `val_time_period`, `test_time_period` can overlap, but they should keep order of `train_time_period` < `val_time_period` < `test_time_period`

    For available configuration options, refer to [here][cesnet_tszoo.configs.time_based_config.TimeBasedConfig--configuration-options].

    Attributes:
        used_train_workers: Tracks the number of train workers in use. Helps determine if the train dataloader should be recreated based on worker changes.
        used_val_workers: Tracks the number of validation workers in use. Helps determine if the validation dataloader should be recreated based on worker changes.
        used_test_workers: Tracks the number of test workers in use. Helps determine if the test dataloader should be recreated based on worker changes.
        used_all_workers: Tracks the total number of all workers in use. Helps determine if the all dataloader should be recreated based on worker changes.
        uses_all_time_period: Whether all time period set should be used.
        import_identifier: Tracks the name of the config upon import. None if not imported.
        logger: Logger for displaying information.     

    The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize] is called:

    Attributes:
        display_train_time_period: Used to display the configured value of `train_time_period`.
        display_val_time_period: Used to display the configured value of `val_time_period`.
        display_test_time_period: Used to display the configured value of `test_time_period`.
        display_all_time_period: Used to display the configured value of `all_time_period`.
        all_time_period: If no specific sets (train/val/test) are provided, all time IDs are used. When any set is defined, only the time IDs in defined sets are used.
        ts_row_ranges: Initialized when `ts_ids` is set. Contains time series IDs in `ts_ids` with their respective time ID ranges (same as `all_time_period`).

        aggregation: The aggregation period used for the data.
        source_type: The source type of the data.
        database_name: Specifies which database this config applies to.
        features_to_take_without_ids: Features to be returned, excluding time or time series IDs.
        indices_of_features_to_take_no_ids: Indices of non-ID features in `features_to_take`.
        ts_id_name: Name of the time series ID, dependent on `source_type`.
        used_singular_train_time_series: Currently used singular train set time series for dataloader.
        used_singular_val_time_series: Currently used singular validation set time series for dataloader.
        used_singular_test_time_series: Currently used singular test set time series for dataloader.
        used_singular_all_time_series: Currently used singular all set time series for dataloader.        
        transformers: Prepared transformers for fitting/transforming. Can be one transformer, array of transformers or `None`.
        train_fillers: Fillers used in the train set. `None` if no filler is used or train set is not used.
        val_fillers: Fillers used in the validation set. `None` if no filler is used or validation set is not used.
        test_fillers: Fillers used in the test set. `None` if no filler is used or test set is not used.
        all_fillers: Fillers used for the all set. `None` if no filler is used or all set is not used.
        anomaly_handlers: Prepared anomaly handlers for fitting/handling anomalies. Can be array of anomaly handlers or `None`.
        is_initialized: Flag indicating if the configuration has already been initialized. If true, config initialization will be skipped.  
        version: Version of cesnet-tszoo this config was made in.
        export_update_needed: Whether config was updated to newer version and should be exported.     

    # Configuration options

    Attributes:
        ts_ids: Defines which time series IDs are used for train/val/test/all. Can be a list of IDs, or an integer/float to specify a random selection. An `int` specifies the number of random time series, and a `float` specifies the proportion of available time series. 
                `int` and `float` must be greater than 0, and a float should be smaller or equal to 1.0.  
        train_time_period: Defines the time period for training set. Can be a range of time IDs or a tuple of datetime objects. Float value is equivalent to percentage of available times with offseted position from previous used set. `Default: None`
        val_time_period: Defines the time period for validation set. Can be a range of time IDs or a tuple of datetime objects. Float value is equivalent to percentage of available times with offseted position from previous used set. `Default: None`
        test_time_period: Defines the time period for test set. Can be a range of time IDs or a tuple of datetime objects. `Default: None`
        features_to_take: Defines which features are used. `Default: "all"`                  
        default_values: Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature. `Default: "default"`
        sliding_window_size: Number of times in one window. Impacts dataloader behavior. Batch sizes affects how much data will be cached for creating windows. `Default: None`
        sliding_window_prediction_size: Number of times to predict from sliding_window_size. Impacts dataloader behavior. Batch sizes affects how much data will be cached for creating windows. `Default: None`
        sliding_window_step: Number of times to move by after each window. `Default: 1`
        set_shared_size: How much times should time periods share. Order of sharing is training set < validation set < test set. Only in effect if sets share less values than set_shared_size. Use float value for percentage of total times or int for count. `Default: 0`
        train_batch_size: Batch size for the train dataloader. Affects number of returned times in one batch. `Default: 32`
        val_batch_size: Batch size for the validation dataloader. Affects number of returned times in one batch. `Default: 64`
        test_batch_size: Batch size for the test dataloader. Affects number of returned times in one batch. `Default: 128`
        all_batch_size: Batch size for the all dataloader. Affects number of returned times in one batch. `Default: 128`   
        fill_missing_with: Defines how to fill missing values in the dataset. Can pass enum [`FillerType`][cesnet_tszoo.utils.enums.FillerType] for built-in filler or pass a type of custom filler that must derive from [`Filler`][cesnet_tszoo.utils.filler.Filler] base class. `Default: None`
        transform_with: Defines the transformer used to transform the dataset. Can pass enum [`TransformerType`][cesnet_tszoo.utils.enums.TransformerType] for built-in transformer, pass a type of custom transformer or instance of already fitted transformer(s). `Default: None`
        handle_anomalies_with: Defines the anomaly handler for handling anomalies in the train set. Can pass enum [`AnomalyHandlerType`][cesnet_tszoo.utils.enums.AnomalyHandlerType] for built-in anomaly handler or a type of custom anomaly handler. `Default: None`
        create_transformer_per_time_series: If `True`, a separate transformer is created for each time series. Not used when using already initialized transformers. `Default: True`
        partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers. `Default: False`
        include_time: If `True`, time data is included in the returned values. `Default: True`
        include_ts_id: If `True`, time series IDs are included in the returned values. `Default: True`
        time_format: Format for the returned time data. When using TimeFormat.DATETIME, time will be returned as separate list along rest of the values. `Default: TimeFormat.ID_TIME`
        train_workers: Number of workers for loading training data. `0` means that the data will be loaded in the main process. `Default: 4`
        val_workers: Number of workers for loading validation data. `0` means that the data will be loaded in the main process. `Default: 3`
        test_workers: Number of workers for loading test data. `0` means that the data will be loaded in the main process. `Default: 2`
        all_workers: Number of workers for loading all data. `0` means that the data will be loaded in the main process. `Default: 4`
        init_workers: Number of workers for initial dataset processing during configuration. `0` means that the data will be loaded in the main process. `Default: 4`
        nan_threshold: Maximum allowable percentage of missing data. Time series exceeding this threshold are excluded. Time series over the threshold will not be used. Used for `train/val/test/all` separately. `Default: 1.0`
        random_state: Fixes randomness for reproducibility during configuration and dataset initialization. `Default: None`                   
    """

    def __init__(self,
                 ts_ids: list[int] | npt.NDArray[np.int_] | float | int,
                 train_time_period: tuple[datetime, datetime] | range | float | None = None,
                 val_time_period: tuple[datetime, datetime] | range | float | None = None,
                 test_time_period: tuple[datetime, datetime] | range | float | None = None,
                 features_to_take: list[str] | Literal["all"] = "all",
                 default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None = "default",
                 sliding_window_size: int | None = None,
                 sliding_window_prediction_size: int | None = None,
                 sliding_window_step: int = 1,
                 set_shared_size: float | int = 0,
                 train_batch_size: int = 32,
                 val_batch_size: int = 64,
                 test_batch_size: int = 128,
                 all_batch_size: int = 128,
                 preprocess_order: list[str, type] = ["handling_anomalies", "filling_gaps", "transforming"],
                 fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None = None,
                 transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None = None,
                 handle_anomalies_with: type | AnomalyHandlerType | Literal["z-score", "interquartile_range"] | None = None,
                 create_transformer_per_time_series: bool = True,
                 partial_fit_initialized_transformers: bool = False,
                 include_time: bool = True,
                 include_ts_id: bool = True,
                 time_format: TimeFormat | Literal["id_time", "datetime", "unix_time", "shifted_unix_time"] = TimeFormat.ID_TIME,
                 train_workers: int = 4,
                 val_workers: int = 3,
                 test_workers: int = 2,
                 all_workers: int = 4,
                 init_workers: int = 4,
                 nan_threshold: float = 1.0,
                 random_state: int | None = None):

        self.ts_ids = ts_ids

        self.ts_row_ranges = None

        self.logger = logging.getLogger("time_config")

        TimeBasedHandler.__init__(self, self.logger, train_batch_size, val_batch_size, test_batch_size, all_batch_size, True, sliding_window_size, sliding_window_prediction_size, sliding_window_step, set_shared_size, train_time_period, val_time_period, test_time_period)
        DatasetConfig.__init__(self, features_to_take, default_values, train_batch_size, val_batch_size, test_batch_size, all_batch_size, preprocess_order, fill_missing_with, transform_with, handle_anomalies_with, partial_fit_initialized_transformers, include_time, include_ts_id, time_format,
                               train_workers, val_workers, test_workers, all_workers, init_workers, nan_threshold, create_transformer_per_time_series, DatasetType.TIME_BASED, DataloaderOrder.SEQUENTIAL, random_state, True, self.logger)

    def _validate_construction(self) -> None:
        """Performs basic parameter validation to ensure correct configuration. More comprehensive validation, which requires dataset-specific data, is handled in [`_dataset_init`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig._dataset_init]. """

        DatasetConfig._validate_construction(self)

        self._validate_set_shared_size_init()
        self._validate_sliding_window_init()
        self._update_batch_sizes(self.train_batch_size, self.val_batch_size, self.test_batch_size, self.all_batch_size)

        assert self.ts_ids is not None, "ts_ids must not be None"

        split_float_total = 0

        if isinstance(self.ts_ids, (float, int)):
            assert self.ts_ids > 0, "ts_ids must be greater than 0"
            if isinstance(self.ts_ids, float):
                split_float_total += self.ts_ids

        # Check if the total of float splits exceeds 1.0
        if split_float_total > 1.0:
            self.logger.error("The total of the float split sizes is greater than 1.0. Current total: %s", split_float_total)
            raise ValueError("Total value of used float split sizes can't be greater than 1.0.")

        self._validate_time_periods_init()

        self.logger.debug("Time-based configuration validated successfully.")

    def _update_batch_sizes(self, train_batch_size: int, val_batch_size: int, test_batch_size: int, all_batch_size: int) -> None:

        # Adjust batch sizes based on sliding_window_size
        if self.sliding_window_size is not None:

            if self.sliding_window_step <= 0:
                raise ValueError("sliding_window_step must be greater or equal to 1.")

            total_window_size = self.sliding_window_size + self.sliding_window_prediction_size

            if isinstance(self.train_batch_size, int) and total_window_size > self.train_batch_size:
                train_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("train_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if isinstance(self.val_batch_size, int) and total_window_size > self.val_batch_size:
                val_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("val_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if isinstance(self.test_batch_size, int) and total_window_size > self.test_batch_size:
                test_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("test_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if isinstance(self.all_batch_size, int) and total_window_size > self.all_batch_size:
                all_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("all_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)

        DatasetConfig._update_batch_sizes(self, train_batch_size, val_batch_size, test_batch_size, all_batch_size)

    def _update_sliding_window(self, sliding_window_size: int | None, sliding_window_prediction_size: int | None, sliding_window_step: int | None, set_shared_size: float | int, all_time_ids: np.ndarray):
        """Updates values related to sliding window. """
        TimeBasedHandler._update_sliding_window(self, sliding_window_size, sliding_window_prediction_size, sliding_window_step, set_shared_size, all_time_ids, self.has_train(), self.has_val(), self.has_test(), self.has_all())

    def _get_train(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the training set. """
        return self.ts_ids, self.train_time_period

    def _get_val(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the validation set. """
        return self.ts_ids, self.val_time_period

    def _get_test(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the test set. """
        return self.ts_ids, self.test_time_period

    def _get_all(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the all set. """
        return self.ts_ids, self.all_time_period

    def has_train(self) -> bool:
        """Returns whether training set is used. """
        return self.train_time_period is not None

    def has_val(self) -> bool:
        """Returns whether validation set is used. """
        return self.val_time_period is not None

    def has_test(self) -> bool:
        """Returns whether test set is used. """
        return self.test_time_period is not None

    def has_all(self) -> bool:
        """Returns whether all set is used. """
        return self.all_time_period is not None

    def _set_time_period(self, all_time_ids: np.ndarray) -> None:
        """Validates and filters `train_time_period`, `val_time_period`, `test_time_period` and `all_time_period` based on `dataset` and `aggregation`. """

        self._prepare_and_set_time_period_sets(all_time_ids, self.time_format)

    def _set_ts(self, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray) -> None:
        """ Validates and filters inputted time series id from `ts_ids` based on `dataset` and `source_type`. Handles random set."""

        random_ts_ids = all_ts_ids[self.ts_id_name]
        random_indices = np.arange(len(all_ts_ids))

        # Process ts_ids if it was specified with times series ids
        if not isinstance(self.ts_ids, (float, int)):
            self.ts_ids, self.ts_row_ranges, _ = SeriesBasedHandler._process_ts_ids(self.ts_ids, all_ts_ids, all_ts_row_ranges, None, None, self.logger, self.ts_id_name, self.random_state)

            mask = np.isin(random_ts_ids, self.ts_ids, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("ts_ids set: %s", self.ts_ids)

        # Convert proportions to total values
        if isinstance(self.ts_ids, float):
            self.ts_ids = int(self.ts_ids * len(random_ts_ids))
            self.logger.debug("ts_ids converted to total values: %s", self.ts_ids)

        # Process random ts_ids if it is to be randomly made
        if isinstance(self.ts_ids, int):
            self.ts_ids, self.ts_row_ranges, random_indices = SeriesBasedHandler._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.ts_ids, random_indices, self.logger, self.ts_id_name, self.random_state)
            self.logger.debug("Random ts_ids set with %s time series.", self.ts_ids)

    def _get_feature_transformers(self) -> np.ndarray[Transformer] | Transformer:
        """Creates transformer/s with `transformer_factory`. """

        if self.transformer_factory.has_already_initialized:
            if not self.has_train() and self.partial_fit_initialized_transformers:
                self.partial_fit_initialized_transformers = False
                self.logger.warning("partial_fit_initialized_transformers will be ignored because train set is not used.")

            transformers = self.transformer_factory.get_already_initialized_transformers()

            if self.transformer_factory.has_single_initialized:
                self.logger.debug("Using already initialized transformer %s.", self.transformer_factory.name)
            else:
                assert len(transformers) == len(self.ts_ids), "Number of time series in ts_ids does not match with number of provided transformers."
                self.create_transformer_per_time_series = True
                self.logger.debug("Using list of initialized transformers of %s", self.transformer_factory.name)
        else:
            if not self.has_train() and not self.transformer_factory.is_empty_factory:
                self.transformer_factory = transformer_factories.get_transformer_factory(None, self.create_transformer_per_time_series, self.partial_fit_initialized_transformers)
                self.logger.warning("No transformer will be used because train set is not used.")

            if self.create_transformer_per_time_series:
                transformers = np.array([self.transformer_factory.create_transformer() for _ in self.ts_ids])
                self.logger.debug("Using list of transformers of %s", self.transformer_factory.name)
            else:
                transformers = self.transformer_factory.create_transformer()
                self.logger.debug("Using transformer %s", self.transformer_factory.name)

        return transformers

    def _get_fillers(self) -> tuple:
        """Creates fillers with `filler_factory`. """

        train_fillers = None
        # Set the fillers for the training set
        if self.has_train():
            train_fillers = np.array([self.filler_factory.create_filler(self.features_to_take_without_ids) for _ in self.ts_ids])
            self.logger.debug("Fillers for training set are set.")

        val_fillers = None
        # Set the fillers for the validation set
        if self.has_val():
            val_fillers = np.array([self.filler_factory.create_filler(self.features_to_take_without_ids) for _ in self.ts_ids])
            self.logger.debug("Fillers for validation set are set.")

        test_fillers = None
        # Set the fillers for the test set
        if self.has_test():
            test_fillers = np.array([self.filler_factory.create_filler(self.features_to_take_without_ids) for _ in self.ts_ids])
            self.logger.debug("Fillers for test set are set.")

        all_fillers = None
        # Set the fillers for the all set
        if self.has_all():
            all_fillers = np.array([self.filler_factory.create_filler(self.features_to_take_without_ids) for _ in self.ts_ids])
            self.logger.debug("Fillers for all set are set.")

        self.logger.debug("Using filler %s", self.filler_factory.name)

        return train_fillers, val_fillers, test_fillers, all_fillers

    def _get_anomaly_handlers(self) -> np.ndarray:
        """Creates anomaly handlers with `anomaly_handler_factory`. """

        if not self.has_train() and not self.anomaly_handler_factory.is_empty_factory:
            self.anomaly_handler_factory = anomaly_handler_factories.get_anomaly_handler_factory(None)
            self.logger.warning("No anomaly handler will be used because train set is not used.")

        anomaly_handlers = np.array([])
        if self.has_train():
            anomaly_handlers = np.array([self.anomaly_handler_factory.create_anomaly_handler() for _ in self.ts_ids])

        self.logger.debug("Using anomaly handler %s", self.anomaly_handler_factory.name)

        return anomaly_handlers

    def _set_per_series_custom_handler(self, factory: PerSeriesCustomHandlerFactory):

        if not self.has_train():
            raise ValueError("To use PerSeriesCustomHandler you need to use train set.")

        handlers = np.array([factory.create_handler() for _ in self.ts_ids])

        self.train_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, False, True, factory.can_apply_to_train, True, PerSeriesCustomHandlerHolder(handlers)))
        self.val_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, True, False, factory.can_apply_to_val and self.has_val(), True, PerSeriesCustomHandlerHolder(handlers)))
        self.test_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, True, False, factory.can_apply_to_test and self.has_test(), True, PerSeriesCustomHandlerHolder(handlers)))
        self.all_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, True, False, factory.can_apply_to_all and self.has_all(), True, PerSeriesCustomHandlerHolder(handlers)))

    def _set_no_fit_custom_handler(self, factory: NoFitCustomHandlerFactory):

        train_handlers = np.array([factory.create_handler() for _ in self.ts_ids]) if self.has_train() else None
        self.train_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, False, False, factory.can_apply_to_train and self.has_train(), True, NoFitCustomHandlerHolder(train_handlers)))

        val_handlers = np.array([factory.create_handler() for _ in self.ts_ids]) if self.has_val() else None
        self.val_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, False, False, factory.can_apply_to_val and self.has_val(), True, NoFitCustomHandlerHolder(val_handlers)))

        test_handlers = np.array([factory.create_handler() for _ in self.ts_ids]) if self.has_test() else None
        self.test_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, False, False, factory.can_apply_to_test and self.has_test(), True, NoFitCustomHandlerHolder(test_handlers)))

        all_handlers = np.array([factory.create_handler() for _ in self.ts_ids]) if self.has_all() else None
        self.all_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, False, False, factory.can_apply_to_all and self.has_all(), True, NoFitCustomHandlerHolder(all_handlers)))

    def _validate_finalization(self) -> None:
        """ Performs final validation of the configuration. Validates whether `train/val/test` are continuos. """

        self._validate_time_periods_overlap()

    def __str__(self) -> str:

        if self.transformer_factory.is_empty_factory:
            transformer_part = f"Transformer type: {self.transformer_factory.name}"
        else:
            transformer_part = f'''Transformer type: {self.transformer_factory.name}
        Is transformer per Time series: {self.create_transformer_per_time_series}
        Are transformers premade: {self.transformer_factory.has_already_initialized}
        Are premade transformers partial_fitted: {self.partial_fit_initialized_transformers}'''

        if self.include_time:
            time_part = f'''Time included: {str(self.include_time)}    
        Time format: {str(self.time_format)}'''
        else:
            time_part = f"Time included: {str(self.include_time)}"

        return f'''
Config Details
    Used for database: {self.database_name}
    Aggregation: {str(self.aggregation)}
    Source: {str(self.source_type)}

    Time series
        Time series IDS: {get_abbreviated_list_string(self.ts_ids)}
    Time periods
        Train time periods: {str(self.display_train_time_period)}
        Val time periods: {str(self.display_val_time_period)}
        Test time periods: {str(self.display_test_time_period)}
        All time periods: {str(self.display_all_time_period)}
    Features
        Taken features: {str(self.features_to_take_without_ids)}
        Default values: {self.default_values}
        Time series ID included: {str(self.include_ts_id)}
        {time_part}
    Sliding window
        Sliding window size: {self.sliding_window_size}
        Sliding window prediction size: {self.sliding_window_prediction_size}
        Sliding window step size: {self.sliding_window_step}
        Set shared size: {self.set_shared_size}
    Fillers
        Filler type: {self.filler_factory.name}
    Transformers
        {transformer_part}
    Anomaly handler
        Anomaly handler type: {self.anomaly_handler_factory.name}        
    Batch sizes
        Train batch size: {self.train_batch_size}
        Val batch size: {self.val_batch_size}
        Test batch size: {self.test_batch_size}
        All batch size: {self.all_batch_size}
    Default workers
        Init worker count: {str(self.init_workers)}
        Train worker count: {str(self.train_workers)}
        Val worker count: {str(self.val_workers)}
        Test worker count: {str(self.test_workers)}
        All worker count: {str(self.all_workers)}
    Other
        Nan threshold: {str(self.nan_threshold)}
        Random state: {self.random_state}
        Version: {self.version}
                '''
