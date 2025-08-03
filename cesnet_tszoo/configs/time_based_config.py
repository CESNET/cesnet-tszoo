from typing import Literal
from datetime import datetime
from numbers import Number

import numpy as np
import numpy.typing as npt

from cesnet_tszoo.utils.filler import filler_from_input_to_type
from cesnet_tszoo.utils.scaler import scaler_from_input_to_scaler_type, Scaler
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME
from cesnet_tszoo.utils.utils import get_abbreviated_list_string
from cesnet_tszoo.utils.enums import FillerType, ScalerType, TimeFormat, DataloaderOrder
from cesnet_tszoo.configs.base_config import DatasetConfig


class TimeBasedConfig(DatasetConfig):
    """
    This class is used for configuring the [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset].

    Used to configure the following:

    - Train, validation, test, test_other, all sets (time period, sizes, features, window size)
    - Handling missing values (default values, [`fillers`][cesnet_tszoo.utils.filler])
    - Data transformation using [`scalers`][cesnet_tszoo.utils.scaler]
    - Dataloader options (train/val/test/all/init workers, batch sizes)
    - Plotting

    **Important Notes:**

    - Custom fillers must inherit from the [`fillers`][cesnet_tszoo.utils.filler.Filler] base class.
    - Fillers can carry over values from the train set to the validation and test sets. For example, [`ForwardFiller`][cesnet_tszoo.utils.filler.ForwardFiller] can carry over values from previous sets.    
    - It is recommended to use the [`scalers`][cesnet_tszoo.utils.scaler.Scaler] base class, though this is not mandatory as long as it meets the required methods.
        - If scalers are already initialized and `create_scaler_per_time_series` is `True` and `partial_fit_initialized_scalers` is `True` then scalers must support `partial_fit`.
        - If `create_scaler_per_time_series` is `True`, scalers must have a `fit` method and `scale_with` should be a list of scalers.
        - If `create_scaler_per_time_series` is `False`, scalers must support `partial_fit`.
        - Scalers must implement the `transform` method.
        - The `fit/partial_fit` and `transform` methods must accept an input of type `np.ndarray` with shape `(times, features)`.
        - Scalers are applied to `test_other` only when `create_scaler_per_time_series` is `False`.    
    - `ts_ids` and `test_ts_ids` must not contain any overlapping time series IDs.
    - `train_time_period`, `val_time_period`, `test_time_period` can overlap, but they should keep order of `train_time_period` < `val_time_period` < `test_time_period`

    For available configuration options, refer to [here][cesnet_tszoo.configs.time_based_config.TimeBasedConfig--configuration-options].

    Attributes:
        used_train_workers: Tracks the number of train workers in use. Helps determine if the train dataloader should be recreated based on worker changes.
        used_val_workers: Tracks the number of validation workers in use. Helps determine if the validation dataloader should be recreated based on worker changes.
        used_test_workers: Tracks the number of test workers in use. Helps determine if the test dataloader should be recreated based on worker changes.
        used_test_other_workers: Tracks the number of test_other workers in use. Helps determine if the test_other dataloader should be recreated based on worker changes.
        used_all_workers: Tracks the total number of all workers in use. Helps determine if the all dataloader should be recreated based on worker changes.
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
        test_ts_row_ranges: Initialized when `test_ts_ids` is set. Contains time series IDs in `test_ts_ids` with their respective time ID ranges (same as `test_time_period`).    
        other_test_fillers: Fillers used in the test_other set. `None` if no filler is used or test_other set is not used.
        has_ts_ids: Flag indicating whether the `ts_ids` is in use.       
        has_test_ts_ids: Flag indicating whether the `test_ts_ids` is in use.    

        aggregation: The aggregation period used for the data.
        source_type: The source type of the data.
        database_name: Specifies which database this config applies to.
        scale_with_display: Used to display the configured type of `scale_with`.
        fill_missing_with_display: Used to display the configured type of `fill_missing_with`.
        features_to_take_without_ids: Features to be returned, excluding time or time series IDs.
        indices_of_features_to_take_no_ids: Indices of non-ID features in `features_to_take`.
        is_scaler_custom: Flag indicating whether the scaler is custom.
        is_filler_custom: Flag indicating whether the filler is custom.
        ts_id_name: Name of the time series ID, dependent on `source_type`.
        used_times: List of all times used in the configuration.
        used_ts_ids: List of all time series IDs used in the configuration.
        used_ts_row_ranges: List of time series IDs with their respective time ID ranges.
        used_fillers: List of all fillers used in the configuration.
        used_singular_train_time_series: Currently used singular train set time series for dataloader.
        used_singular_val_time_series: Currently used singular validation set time series for dataloader.
        used_singular_test_time_series: Currently used singular test set time series for dataloader.
        used_singular_test_other_time_series: Currently used singular test other set time series for dataloader.
        used_singular_all_time_series: Currently used singular all set time series for dataloader.             
        scalers: Prepared scalers for fitting/transforming. Can be one scaler, array of scalers or `None`.
        are_scalers_premade: Indicates whether the scalers are premade.
        has_train: Flag indicating whether the training set is in use.
        has_val: Flag indicating whether the validation set is in use.
        has_test: Flag indicating whether the test set is in use.
        has_all: Flag indicating whether the all set is in use.
        train_fillers: Fillers used in the train set. `None` if no filler is used or train set is not used.
        val_fillers: Fillers used in the validation set. `None` if no filler is used or validation set is not used.
        test_fillers: Fillers used in the test set. `None` if no filler is used or test set is not used.
        all_fillers: Fillers used for the all set. `None` if no filler is used or all set is not used.
        is_initialized: Flag indicating if the configuration has already been initialized. If true, config initialization will be skipped.      

    # Configuration options

    Attributes:
        ts_ids: Defines which time series IDs are used for train/val/test/all. Can be a list of IDs, or an integer/float to specify a random selection. An `int` specifies the number of random time series, and a `float` specifies the proportion of available time series. 
                `int` and `float` must be greater than 0, and a float should be smaller or equal to 1.0. Using `int` or `float` guarantees that no time series from `test_ts_ids` will be used. `Default: None`    
        train_time_period: Defines the time period for training set. Can be a range of time IDs or a tuple of datetime objects. Float value is equivalent to percentage of available times with offseted position from previous used set. `Default: None`
        val_time_period: Defines the time period for validation set. Can be a range of time IDs or a tuple of datetime objects. Float value is equivalent to percentage of available times with offseted position from previous used set. `Default: None`
        test_time_period: Defines the time period for test set. Can be a range of time IDs or a tuple of datetime objects. `Default: None`
        features_to_take: Defines which features are used. `Default: "all"` 
        test_ts_ids: Defines which time series IDs are used in the test_other set. Same as `ts_ids` but for the test_other set. These time series only use times in `test_time_period`. `Default: None`                   
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
        scale_with: Defines the scaler used to transform the dataset. Can pass enum [`ScalerType`][cesnet_tszoo.utils.enums.ScalerType] for built-in scaler, pass a type of custom scaler or instance of already fitted scaler(s). `Default: None`
        create_scaler_per_time_series: If `True`, a separate scaler is created for each time series. Not used when using already initialized scalers. `Default: True`
        partial_fit_initialized_scalers: If `True`, partial fitting on train set is performed when using initiliazed scalers. `Default: False`
        include_time: If `True`, time data is included in the returned values. `Default: True`
        include_ts_id: If `True`, time series IDs are included in the returned values. `Default: True`
        time_format: Format for the returned time data. When using TimeFormat.DATETIME, time will be returned as separate list along rest of the values. `Default: TimeFormat.ID_TIME`
        train_workers: Number of workers for loading training data. `0` means that the data will be loaded in the main process. `Default: 4`
        val_workers: Number of workers for loading validation data. `0` means that the data will be loaded in the main process. `Default: 3`
        test_workers: Number of workers for loading test and test_other data. `0` means that the data will be loaded in the main process. `Default: 2`
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
                 test_ts_ids: list[int] | npt.NDArray[np.int_] | float | int | None = None,
                 default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None = "default",
                 sliding_window_size: int | None = None,
                 sliding_window_prediction_size: int | None = None,
                 sliding_window_step: int = 1,
                 set_shared_size: float | int = 0,
                 train_batch_size: int = 32,
                 val_batch_size: int = 64,
                 test_batch_size: int = 128,
                 all_batch_size: int = 128,
                 fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None = None,
                 scale_with: type | list[Scaler] | np.ndarray[Scaler] | ScalerType | Scaler | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_scaler", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None = None,
                 create_scaler_per_time_series: bool = True,
                 partial_fit_initialized_scalers: bool = False,
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
        self.train_time_period = train_time_period
        self.val_time_period = val_time_period
        self.test_time_period = test_time_period
        self.test_ts_ids = test_ts_ids

        self.display_train_time_period = None
        self.display_val_time_period = None
        self.display_test_time_period = None
        self.display_all_time_period = None
        self.all_time_period = None
        self.used_test_other_workers = None
        self.ts_row_ranges = None
        self.test_ts_row_ranges = None
        self.other_test_fillers = None
        self.has_ts_ids = False
        self.has_test_ts_ids = False
        self.used_singular_test_other_time_series = None

        super(TimeBasedConfig, self).__init__(features_to_take, default_values, sliding_window_size, sliding_window_prediction_size, sliding_window_step, set_shared_size, train_batch_size, val_batch_size, test_batch_size, all_batch_size, fill_missing_with, scale_with, partial_fit_initialized_scalers, include_time, include_ts_id, time_format,
                                              train_workers, val_workers, test_workers, all_workers, init_workers, nan_threshold, create_scaler_per_time_series, False, DataloaderOrder.SEQUENTIAL, random_state)

    def _validate_construction(self) -> None:
        """Performs basic parameter validation to ensure correct configuration. More comprehensive validation, which requires dataset-specific data, is handled in [`_dataset_init`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig._dataset_init]. """

        super(TimeBasedConfig, self)._validate_construction()

        assert self.ts_ids is not None, "ts_ids must not be None"

        if self.test_time_period is None and self.test_ts_ids is not None:
            self.test_ts_ids = None
            self.logger.warning("test_ts_ids has been ignored because test_time_period is set to None.")

        split_float_total = 0

        if isinstance(self.ts_ids, (float, int)):
            assert self.ts_ids > 0, "ts_ids must be greater than 0"
            if isinstance(self.ts_ids, float):
                split_float_total += self.ts_ids

        if isinstance(self.test_ts_ids, (float, int)):
            assert self.test_ts_ids > 0, "test_ts_ids must be greater than 0"
            if isinstance(self.test_ts_ids, float):
                split_float_total += self.test_ts_ids

        # Check if the total of float splits exceeds 1.0
        if split_float_total > 1.0:
            self.logger.error("The total of the float split sizes is greater than 1.0. Current total: %s", split_float_total)
            raise ValueError("Total value of used float split sizes can't be greater than 1.0.")

        split_time_float_total = 0
        train_used_float = None if self.train_time_period is None else False
        val_used_float = None if self.val_time_period is None else False

        if isinstance(self.train_time_period, (float, int)):
            self.train_time_period = float(self.train_time_period)
            assert self.train_time_period > 0.0, "train_time_period must be greater than 0"
            split_time_float_total += self.train_time_period
            train_used_float = True

        if isinstance(self.val_time_period, (float, int)):
            if train_used_float is False:
                raise ValueError("val_time_period cant use float to be set, because train_time_period was set, but did not use float.")
            self.val_time_period = float(self.val_time_period)
            assert self.val_time_period > 0.0, "val_time_period must be greater than 0"
            split_time_float_total += self.val_time_period
            val_used_float = True

        if isinstance(self.test_time_period, (float, int)):
            if train_used_float is False or val_used_float is False:
                raise ValueError("test_time_period cant use float to be set, because previous periods were set, but did not use float.")
            self.test_time_period = float(self.test_time_period)
            assert self.test_time_period > 0.0, "test_time_period must be greater than 0"
            split_time_float_total += self.test_time_period

        # Check if the total of float splits exceeds 1.0
        if split_time_float_total > 1.0:
            self.logger.error("The total of the float split sizes for time periods is greater than 1.0. Current total: %s", split_time_float_total)
            raise ValueError("Total value of used float split sizes for time periods can't be greater than 1.0.")

        self.logger.debug("Time-based configuration validated successfully.")

    def _update_sliding_window(self, sliding_window_size: int | None, sliding_window_prediction_size: int | None, sliding_window_step: int | None, set_shared_size: float | int, all_time_ids: np.ndarray):
        if isinstance(set_shared_size, float):
            assert set_shared_size >= 0 and set_shared_size <= 1, "set_shared_size float value must be between or equal to 0 and 1."
            set_shared_size = int(len(all_time_ids) * set_shared_size)

        assert set_shared_size >= 0, "set_shared_size must be of positive value."

        # Ensure sliding_window_size is either None or a valid integer greater than 1
        assert sliding_window_size is None or (isinstance(sliding_window_size, int) and sliding_window_size > 1), "sliding_window_size must be an integer greater than 1, or None."

        # Ensure sliding_window_prediction_size is either None or a valid integer greater or equal to 1
        assert sliding_window_prediction_size is None or (isinstance(sliding_window_prediction_size, int) and sliding_window_prediction_size >= 1), "sliding_window_prediction_size must be an integer greater than 1, or None."

        # Both sliding_window_size and sliding_window_prediction_size must be set or None
        assert (sliding_window_size is None and sliding_window_prediction_size is None) or (sliding_window_size is not None and sliding_window_prediction_size is not None), "Both sliding_window_size and sliding_window_prediction_size must be set or None."

        # Adjust batch sizes based on sliding_window_size
        if sliding_window_size is not None:

            if sliding_window_step <= 0:
                raise ValueError("sliding_window_step must be greater or equal to 1.")

            if set_shared_size == self.set_shared_size:
                if self.has_train and len(self.train_time_period) < sliding_window_size + sliding_window_prediction_size:
                    raise ValueError("New sliding window size + prediction size is larger than the number of times in train_time_period.")

                if self.has_val and len(self.val_time_period) < sliding_window_size + sliding_window_prediction_size:
                    raise ValueError("New sliding window size + prediction size is larger than the number of times in val_time_period.")

                if self.has_test and len(self.test_time_period) < sliding_window_size + sliding_window_prediction_size:
                    raise ValueError("New sliding window size + prediction size is larger than the number of times in test_time_period.")

                if self.has_all and len(self.all_time_period) < sliding_window_size + sliding_window_prediction_size:
                    raise ValueError("New sliding window size + prediction size is larger than the number of times in all_time_period.")

            total_window_size = sliding_window_size + sliding_window_prediction_size

            if total_window_size > self.train_batch_size:
                self.train_batch_size = sliding_window_size + sliding_window_prediction_size
                self.logger.info("train_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > self.val_batch_size:
                self.val_batch_size = sliding_window_size + sliding_window_prediction_size
                self.logger.info("val_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > self.test_batch_size:
                self.test_batch_size = sliding_window_size + sliding_window_prediction_size
                self.logger.info("test_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > self.all_batch_size:
                self.all_batch_size = sliding_window_size + sliding_window_prediction_size
                self.logger.info("all_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)

        self.sliding_window_size = sliding_window_size
        self.sliding_window_prediction_size = sliding_window_prediction_size
        self.sliding_window_step = sliding_window_step
        self.set_shared_size = set_shared_size

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

    def _get_test_other(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the test_other set. """
        return self.test_ts_ids, self.test_time_period

    def _set_time_period(self, all_time_ids: np.ndarray) -> None:
        """Validates and filters `train_time_period`, `val_time_period`, `test_time_period` and `all_time_period` based on `dataset` and `aggregation`. """

        if isinstance(self.set_shared_size, float):
            self.set_shared_size = int(len(all_time_ids) * self.set_shared_size)
            self.logger.debug("Converted set_shared_size from float to int value.")

        times_to_share = None

        # Used when periods are set with float
        start = 0
        end = len(all_time_ids)

        if isinstance(self.train_time_period, float):
            offset_from_start = int(end * self.train_time_period)
            self.train_time_period = range(start, start + offset_from_start)
            start += offset_from_start
            self.logger.debug("train_time_period set with float value. Using range: %s", self.train_time_period)

        # Process and validate train time period
        self.train_time_period, self.display_train_time_period = self._process_time_period(self.train_time_period, all_time_ids, times_to_share)
        self.has_train = self.train_time_period is not None

        if self.has_train:
            if self.sliding_window_size is not None and len(self.train_time_period) < self.sliding_window_size + self.sliding_window_prediction_size:
                raise ValueError("Sliding window size + prediction size is larger than the number of times in train_time_period.")
            self.logger.debug("Processed train_time_period: %s, display_train_time_period: %s", self.train_time_period, self.display_train_time_period)
            if self.set_shared_size > 0:
                if self.set_shared_size >= len(self.train_time_period):
                    times_to_share = self.train_time_period[0: len(self.train_time_period)]
                    times_to_share = all_time_ids[times_to_share[ID_TIME_COLUMN_NAME]]
                    self.logger.warning("Whole training set will be shared to the next set. Consider increasing train_time_period or lowering set_shared_size. Current set_shared_size in count value is %s", self.set_shared_size)
                else:
                    times_to_share = self.train_time_period[-self.set_shared_size:len(self.train_time_period)]
                    times_to_share = all_time_ids[times_to_share[ID_TIME_COLUMN_NAME]]

        if isinstance(self.val_time_period, float):
            offset_from_start = int(end * self.val_time_period)
            self.val_time_period = range(start, start + offset_from_start)
            start += offset_from_start
            self.logger.debug("val_time_period set with float value. Using range: %s", self.val_time_period)

        # Process and validate validation time period
        self.val_time_period, self.display_val_time_period = self._process_time_period(self.val_time_period, all_time_ids, times_to_share)
        self.has_val = self.val_time_period is not None

        if self.has_val:
            if self.sliding_window_size is not None and len(self.val_time_period) < self.sliding_window_size + self.sliding_window_prediction_size:
                raise ValueError("Sliding window size + prediction size is larger than the number of times in val_time_period.")
            self.logger.debug("Processed val_time_period: %s, display_val_time_period: %s", self.val_time_period, self.display_val_time_period)
            if self.set_shared_size > 0:
                if self.set_shared_size >= len(self.val_time_period):
                    times_to_share = self.val_time_period[0: len(self.val_time_period)]
                    times_to_share = all_time_ids[times_to_share[ID_TIME_COLUMN_NAME]]
                    self.logger.warning("Whole validation set will be shared to the next set. Consider increasing val_time_period or lowering set_shared_size. Current set_shared_size in count value is %s", self.set_shared_size)
                else:
                    times_to_share = self.val_time_period[-self.set_shared_size:len(self.val_time_period)]
                    times_to_share = all_time_ids[times_to_share[ID_TIME_COLUMN_NAME]]

        if isinstance(self.test_time_period, float):
            offset_from_start = int(end * self.test_time_period)
            self.test_time_period = range(start, start + offset_from_start)
            start += offset_from_start
            self.logger.debug("test_time_period set with float value. Using range: %s", self.test_time_period)

        # Process and validate test time period
        self.test_time_period, self.display_test_time_period = self._process_time_period(self.test_time_period, all_time_ids, times_to_share)
        self.has_test = self.test_time_period is not None

        if self.has_test:
            if self.sliding_window_size is not None and len(self.test_time_period) < self.sliding_window_size + self.sliding_window_prediction_size:
                raise ValueError("Sliding window size + prediction size is larger than the number of times in test_time_period.")
            self.logger.debug("Processed test_time_period: %s, display_test_time_period: %s", self.test_time_period, self.display_test_time_period)

        if not self.has_train and not self.has_val and not self.has_test:
            self.all_time_period = all_time_ids.copy()
            self.all_time_period = self._set_time_period_form(self.all_time_period, all_time_ids)
            self.logger.info("Using all times for all_time_period because train_time_period, val_time_period, and test_time_period are all set to None.")
        else:
            for temp_time_period in [self.train_time_period, self.val_time_period, self.test_time_period]:
                if temp_time_period is None:
                    continue
                elif self.all_time_period is None:
                    self.all_time_period = temp_time_period.copy()
                else:
                    self.all_time_period = np.concatenate((self.all_time_period, temp_time_period))

            if self.has_train:
                self.logger.debug("all_time_period includes values from train_time_period.")
            if self.has_val:
                self.logger.debug("all_time_period includes values from val_time_period.")
            if self.has_test:
                self.logger.debug("all_time_period includes values from test_time_period.")

            self.all_time_period = np.unique(self.all_time_period)

        self.has_all = self.all_time_period is not None

        if self.has_all:
            self.display_all_time_period = range(self.all_time_period[ID_TIME_COLUMN_NAME][0], self.all_time_period[ID_TIME_COLUMN_NAME][-1] + 1)
            if self.sliding_window_size is not None and len(self.all_time_period) < self.sliding_window_size + self.sliding_window_prediction_size:
                raise ValueError("Sliding window size + prediction size is larger than the number of times in all_time_period.")
            self.logger.debug("Processed all_time_period: %s, display_all_time_period: %s", self.all_time_period, self.display_all_time_period)

    def _set_ts(self, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray) -> None:
        """ Validates and filters inputted time series id from `ts_ids` and `test_ts_ids` based on `dataset` and `source_type`. Handles random set."""

        random_ts_ids = all_ts_ids[self.ts_id_name]
        random_indices = np.arange(len(all_ts_ids))

        # Process ts_ids if it was specified with times series ids
        if not isinstance(self.ts_ids, (float, int)):
            self.ts_ids, self.ts_row_ranges, _ = self._process_ts_ids(self.ts_ids, all_ts_ids, all_ts_row_ranges, None, None)
            self.has_ts_ids = True

            mask = np.isin(random_ts_ids, self.ts_ids, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("ts_ids set: %s", self.ts_ids)

        # Process test_ts_ids if it was specified with times series ids
        if self.test_ts_ids is not None and not isinstance(self.test_ts_ids, (float, int)):
            self.test_ts_ids, self.test_ts_row_ranges, _ = self._process_ts_ids(self.test_ts_ids, all_ts_ids, all_ts_row_ranges, None, None)
            self.has_test_ts_ids = True

            mask = np.isin(random_ts_ids, self.test_ts_ids, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("test_ts_ids set: %s", self.test_ts_ids)

        # Convert proportions to total values
        if isinstance(self.ts_ids, float):
            self.ts_ids = int(self.ts_ids * len(random_ts_ids))
            self.logger.debug("ts_ids converted to total values: %s", self.ts_ids)
        if isinstance(self.test_ts_ids, float):
            self.test_ts_ids = int(self.test_ts_ids * len(random_ts_ids))
            self.logger.debug("test_ts_ids converted to total values: %s", self.test_ts_ids)

        # Process random ts_ids if it is to be randomly made
        if isinstance(self.ts_ids, int):
            self.ts_ids, self.ts_row_ranges, random_indices = self._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.ts_ids, random_indices)
            self.has_ts_ids = True
            self.logger.debug("Random ts_ids set with %s time series.", self.ts_ids)

        # Process random test_ts_ids if it is to be randomly made
        if isinstance(self.test_ts_ids, int):
            self.test_ts_ids, self.test_ts_row_ranges, random_indices = self._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.test_ts_ids, random_indices)
            self.has_test_ts_ids = True
            self.logger.debug("Random test_ts_ids set with %s time series.", self.test_ts_ids)

    def _set_feature_scalers(self) -> None:
        """Creates and/or validates scalers based on the `scale_with` parameter. """

        if self.scale_with is None:
            self.scale_with_display = None
            self.are_scalers_premade = False
            self.scalers = None
            self.is_scaler_custom = None

            self.logger.debug("No scaler will be used because scale_with is not set.")
            return

        if not self.has_train:
            if self.partial_fit_initialized_scalers:
                self.logger.warning("partial_fit_initialized_scalers will be ignored because train set is not used.")
            self.partial_fit_initialized_scalers = False

        # Treat scale_with as a list of initialized scalers
        if isinstance(self.scale_with, (list, np.ndarray)):
            self.create_scaler_per_time_series = True

            self.scalers = np.array(self.scale_with)
            self.scale_with = None

            assert len(self.scalers) == len(self.ts_ids), "Number of time series in ts_ids does not match with number of provided scalers."

            # Ensure that all scalers in the list are of the same type
            for scaler in self.scalers:
                if isinstance(scaler, (type, ScalerType)):
                    raise ValueError("scaler_with as a list of scalers must contain only initialized scalers.")

                new_scale_with, self.scale_with_display = scaler_from_input_to_scaler_type(type(scaler), check_for_fit=False, check_for_partial_fit=self.partial_fit_initialized_scalers)

                if self.scale_with is None:
                    self.scale_with = new_scale_with
                elif self.scale_with != new_scale_with:
                    raise ValueError("Scalers in scale_with must all be of the same type.")

            self.are_scalers_premade = True

            self.is_scaler_custom = "Custom" in self.scale_with_display
            self.logger.debug("Using list of initialized scalers of type: %s", self.scale_with_display)

        # Treat scale_with as already initialized scaler
        elif not isinstance(self.scale_with, (type, ScalerType)):
            self.create_scaler_per_time_series = False

            self.scalers = self.scale_with

            self.scale_with, self.scale_with_display = scaler_from_input_to_scaler_type(type(self.scale_with), check_for_fit=False, check_for_partial_fit=self.partial_fit_initialized_scalers)

            self.are_scalers_premade = True

            self.is_scaler_custom = "Custom" in self.scale_with_display
            self.logger.debug("Using initialized scaler of type: %s", self.scale_with_display)

        # Treat scale_with as uninitialized scaler
        else:
            if not self.has_train:
                self.scale_with = None
                self.scale_with_display = None
                self.are_scalers_premade = False
                self.scalers = None
                self.is_scaler_custom = None

                self.logger.warning("No scaler will be used because train set is not used.")
                return

            self.scale_with, self.scale_with_display = scaler_from_input_to_scaler_type(self.scale_with, check_for_fit=self.create_scaler_per_time_series, check_for_partial_fit=not self.create_scaler_per_time_series)

            self.are_scalers_premade = False

            self.is_scaler_custom = "Custom" in self.scale_with_display
            if self.create_scaler_per_time_series:
                self.scalers = np.array([self.scale_with() for _ in self.ts_ids])
                self.logger.debug("Using list of uninitialized scalers of type: %s", self.scale_with_display)
            else:
                self.scalers = self.scale_with()
                self.logger.debug("Using uninitialized scaler of type: %s", self.scale_with_display)

    def _set_fillers(self) -> None:
        """Creates and/or validates fillers based on the `fill_missing_with` parameter. """

        self.fill_missing_with, self.fill_missing_with_display = filler_from_input_to_type(self.fill_missing_with)
        self.is_filler_custom = "Custom" in self.fill_missing_with_display if self.fill_missing_with is not None else None

        if self.fill_missing_with is None:
            self.logger.debug("No filler is used because fill_missing_with is set to None.")
            return

        # Set the fillers for the training set
        if self.has_train:
            self.train_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.ts_ids])
            self.logger.debug("Fillers for training set are set.")

        # Set the fillers for the validation set
        if self.has_val:
            self.val_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.ts_ids])
            self.logger.debug("Fillers for validation set are set.")

        # Set the fillers for the test set
        if self.has_test:
            self.test_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.ts_ids])
            self.logger.debug("Fillers for test set are set.")

        # Set the fillers for the all set
        if self.has_all:
            self.all_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.ts_ids])
            self.logger.debug("Fillers for all set are set.")

        # Set the fillers for the test_other set
        if self.has_test_ts_ids:
            self.other_test_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.test_ts_ids])
            self.logger.debug("Fillers for other_test set are set.")

    def _validate_finalization(self) -> None:
        """ Performs final validation of the configuration. Validates if `train/val/test` are continuos and that there are no overlapping time series ids in `ts_ids` and `test_ts_ids`."""

        previous_first_time_id = None
        previous_last_time_id = None

        # Validates if time periods are continuos
        for time_period in [self.train_time_period, self.val_time_period, self.test_time_period]:
            if time_period is None:
                continue

            current_first_time_id = time_period[0][ID_TIME_COLUMN_NAME]
            current_last_time_id = time_period[-1][ID_TIME_COLUMN_NAME]

            # Check if the first time ID is valid in relation to the previous time period's first time ID
            if previous_first_time_id is not None:
                if current_first_time_id < previous_first_time_id:
                    self.logger.error("Starting time ids of train/val/test must follow this rule: train < val < test")
                    raise ValueError(f"Starting time ids of train/val/test must follow this rule: train < val < test. "
                                     f"Current first time ID: {current_first_time_id}, previous first time ID: {previous_first_time_id}")

                if current_first_time_id > previous_last_time_id + 1:
                    self.logger.error("Starting time ids of train/val/test must be smaller or equal to last_id(next_split) + 1")
                    raise ValueError(f"Starting time ids of train/val/test must be smaller or equal to last_id(next_split) + 1. "
                                     f"Current first time ID: {current_first_time_id}, previous last time ID: {previous_last_time_id}")

            # Check if the last time ID is valid in relation to the previous time period's last time ID
            if previous_last_time_id is not None:
                if current_last_time_id < previous_last_time_id:
                    self.logger.error("Last time ids of train/val/test must be equal or larger than last_id(next_split)")
                    raise ValueError(f"Last time ids of train/val/test must be equal or larger than last_id(next_split). "
                                     f"Current last time ID: {current_last_time_id}, previous last time ID: {previous_last_time_id}")

            previous_first_time_id = current_first_time_id
            previous_last_time_id = current_last_time_id

        if self.scale_with is not None and self.create_scaler_per_time_series and self.test_ts_ids is not None:
            self.logger.warning("Scalers won't be used on time series in test_ts_ids, if create_scaler_per_time_series is true.")

        # Check for overlap between ts_ids and test_ts_ids
        if self.ts_ids is not None and self.test_ts_ids is not None:
            mask = np.isin(self.ts_ids, self.test_ts_ids)
            if len(self.ts_ids[mask]) > 0:
                self.logger.error("ts_ids and test_ts_ids can't have the same IDs!")
                raise ValueError(f"ts_ids and test_ts_ids can't have the same IDs. Overlapping IDs: {self.ts_ids[mask]}")

    def _try_update_version(self):
        super()._try_update_version()

    def __str__(self) -> str:

        if self.scale_with is None:
            scaler_part = f"Scaler type: {str(self.scale_with_display)}"
        else:
            scaler_part = f'''Scaler type: {str(self.scale_with_display)}
        Is scaler per Time series: {self.create_scaler_per_time_series}
        Are scalers premade: {self.are_scalers_premade}
        Are premade scalers partial_fitted: {self.partial_fit_initialized_scalers}'''

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
        Test time series IDS: {get_abbreviated_list_string(self.test_ts_ids)}
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
        Filler type: {str(self.fill_missing_with_display)}
    Scalers
        {scaler_part}
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
