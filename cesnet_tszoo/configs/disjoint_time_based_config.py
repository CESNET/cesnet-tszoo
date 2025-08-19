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


class DisjointTimeBasedConfig(SeriesBasedHandler, TimeBasedHandler, DatasetConfig):

    def __init__(self,
                 train_ts: list[int] | npt.NDArray[np.int_] | float | int | None,
                 val_ts: list[int] | npt.NDArray[np.int_] | float | int | None,
                 test_ts: list[int] | npt.NDArray[np.int_] | float | int | None,
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
                 fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None = None,
                 transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None = None,
                 partial_fit_initialized_transformer: bool = False,
                 include_time: bool = True,
                 include_ts_id: bool = True,
                 time_format: TimeFormat | Literal["id_time", "datetime", "unix_time", "shifted_unix_time"] = TimeFormat.ID_TIME,
                 train_workers: int = 4,
                 val_workers: int = 3,
                 test_workers: int = 2,
                 init_workers: int = 4,
                 nan_threshold: float = 1.0,
                 random_state: int | None = None):

        self.logger = logging.getLogger("combined_config")

        TimeBasedHandler.__init__(self, self.logger, train_batch_size, val_batch_size, test_batch_size, 1, True, sliding_window_size, sliding_window_prediction_size, sliding_window_step, set_shared_size, train_time_period, val_time_period, test_time_period)
        SeriesBasedHandler.__init__(self, self.logger, True, train_ts, val_ts, test_ts)
        DatasetConfig.__init__(self, features_to_take, default_values, train_batch_size, val_batch_size, test_batch_size, 1, fill_missing_with, transform_with, partial_fit_initialized_transformer, include_time, include_ts_id, time_format,
                               train_workers, val_workers, test_workers, 1, init_workers, nan_threshold, False, DatasetType.DISJOINT_TIME_BASED, DataloaderOrder.SEQUENTIAL, random_state, self.logger)

    def _validate_construction(self) -> None:
        """Performs basic parameter validation to ensure correct configuration. More comprehensive validation, which requires dataset-specific data, is handled in [`_dataset_init`][cesnet_tszoo.configs.multi_time_based_config.MultiTimeBasedConfig._dataset_init]. """

        DatasetConfig._validate_construction(self)

        if self.train_ts is None or self.train_time_period is None:
            if self.train_ts is not None:
                self.logger.error("When train_ts is not None you must set train_time_period or set train_ts as None.")
                raise ValueError("When train_ts is not None you must set train_time_period or set train_ts as None.")
            if self.train_time_period is not None:
                self.logger.error("When train_time_period is not None you must set train_ts or set train_time_period as None.")
                raise ValueError("When train_time_period is not None you must set train_ts or set train_time_period as None.")

        if self.val_ts is None or self.val_time_period is None:
            if self.val_ts is not None:
                self.logger.error("When val_ts is not None you must set val_time_period or set val_ts as None.")
                raise ValueError("When val_ts is not None you must set val_time_period or set val_ts as None.")
            if self.val_time_period is not None:
                self.logger.error("When val_time_period is not None you must set val_ts or set val_time_period as None.")
                raise ValueError("When val_time_period is not None you must set val_ts or set val_time_period as None.")

        if self.test_ts is None or self.test_time_period is None:
            if self.test_ts is not None:
                self.logger.error("When test_ts is not None you must set test_time_period or set test_ts as None.")
                raise ValueError("When test_ts is not None you must set test_time_period or set test_ts as None.")
            if self.test_time_period is not None:
                self.logger.error("When test_time_period is not None you must set test_ts or set test_time_period as None.")
                raise ValueError("When test_time_period is not None you must set test_ts or set test_time_period as None.")

        if self.train_ts is None and self.val_ts is None and self.test_ts is None:
            self.logger.error("No set for time series has been set. You must set at least one time series set and its respective time period.")
            raise ValueError("No set for time series has been set. You must set at least one time series set and its respective time period.")

        self._validate_time_periods_init()
        self._validate_ts_init()
        self._validate_set_shared_size_init()
        self._validate_sliding_window_init()
        self._update_batch_sizes(self.train_batch_size, self.val_batch_size, self.test_batch_size, self.all_batch_size)

        self.logger.debug("Time-based configuration validated successfully.")

    def _update_batch_sizes(self, train_batch_size: int, val_batch_size: int, test_batch_size: int, all_batch_size: int) -> None:

        # Adjust batch sizes based on sliding_window_size
        if self.sliding_window_size is not None:

            if self.sliding_window_step <= 0:
                raise ValueError("sliding_window_step must be greater or equal to 1.")

            total_window_size = self.sliding_window_size + self.sliding_window_prediction_size

            if isinstance(self.train_batch_size, int) and total_window_size > self.train_batch_size:
                self.train_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("train_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if isinstance(self.val_batch_size, int) and total_window_size > self.val_batch_size:
                self.val_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("val_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if isinstance(self.test_batch_size, int) and total_window_size > self.test_batch_size:
                self.test_batch_size = self.sliding_window_size + self.sliding_window_prediction_size
                self.logger.info("test_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)

        DatasetConfig._update_batch_sizes(self, train_batch_size, val_batch_size, test_batch_size, all_batch_size)

    def _update_sliding_window(self, sliding_window_size: int | None, sliding_window_prediction_size: int | None, sliding_window_step: int | None, set_shared_size: float | int, all_time_ids: np.ndarray):
        TimeBasedHandler._update_sliding_window(self, sliding_window_size, sliding_window_prediction_size, sliding_window_step, set_shared_size, all_time_ids, self.has_train(), self.has_val(), self.has_test(), self.has_all())

    def _get_train(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the training set. """
        return self.train_ts, self.train_time_period

    def _get_val(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the validation set. """
        return self.val_ts, self.val_time_period

    def _get_test(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the test set. """
        return self.test_ts, self.test_time_period

    def _get_all(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the all set. """
        return None, None

    def has_train(self) -> bool:
        """Returns whether training set is used. """
        return self.train_ts is not None and self.train_time_period is not None

    def has_val(self) -> bool:
        """Returns whether validation set is used. """
        return self.val_ts is not None and self.val_time_period is not None

    def has_test(self) -> bool:
        """Returns whether test set is used. """
        return self.test_ts is not None and self.test_time_period is not None

    def has_all(self) -> bool:
        """Returns whether all set is used. """
        return False

    def _set_time_period(self, all_time_ids: np.ndarray) -> None:
        """Validates and filters `train_time_period`, `val_time_period`, `test_time_period` and `all_time_period` based on `dataset` and `aggregation`. """

        self._prepare_and_set_time_period_sets(all_time_ids, self.time_format)

    def _set_ts(self, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray) -> None:
        """ Validates and filters inputted time series id from `ts_ids` and `test_ts_ids` based on `dataset` and `source_type`. Handles random set."""

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

        if not self.has_train():
            if self.partial_fit_initialized_transformers:
                self.logger.warning("partial_fit_initialized_transformers will be ignored because train set is not used.")
            self.partial_fit_initialized_transformers = False

        # Treat transform_with as already initialized transformer
        if not isinstance(self.transform_with, (type, TransformerType)):

            self.transformers = self.transform_with

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

            self.transform_with, self.transform_with_display = transformer_from_input_to_transformer_type(self.transform_with, check_for_fit=self.create_transformer_per_time_series, check_for_partial_fit=not self.create_transformer_per_time_series)

            self.are_transformers_premade = False

            self.is_transformer_custom = "Custom" in self.transform_with_display
            self.transformers = self.transform_with()
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
        self.all_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.all_ts])
        self.logger.debug("Fillers for all set are set.")

    def _validate_finalization(self) -> None:
        """ Performs final validation of the configuration. Validates whether `train/val/test` are continuos."""

        self._validate_time_periods_overlap()
        # self._validate_ts_overlap()

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
Config Details
    Used for database: {self.database_name}
    Aggregation: {str(self.aggregation)}
    Source: {str(self.source_type)}

    Time series
        Train time series IDs: {get_abbreviated_list_string(self.train_ts)}
        Val time series IDs: {get_abbreviated_list_string(self.val_ts)}
        Test time series IDs: {get_abbreviated_list_string(self.test_ts)}
    Time periods
        Train time periods: {str(self.display_train_time_period)}
        Val time periods: {str(self.display_val_time_period)}
        Test time periods: {str(self.display_test_time_period)}
    Features
        Taken features: {str(self.features_to_take_without_ids)}
        Default values: {self.default_values}
        Time series ID included: {str(self.include_ts_id)}
        {time_part}
    Sliding window
        Sliding window size: {self.sliding_window_size}
        Sliding window prediction size: {self.sliding_window_prediction_size}
        Sliding window step size: {self.sliding_window_step}
    Fillers
        Filler type: {str(self.fill_missing_with_display)}
    Transformers
        {transformer_part}
    Batch sizes
        Train batch size: {self.train_batch_size}
        Val batch size: {self.val_batch_size}
        Test batch size: {self.test_batch_size}
    Default workers
        Init worker count: {str(self.init_workers)}
        Train worker count: {str(self.train_workers)}
        Val worker count: {str(self.val_workers)}
        Test worker count: {str(self.test_workers)}
    Other
        Nan threshold: {str(self.nan_threshold)}
        Random state: {self.random_state}
        Version: {self.version}
                '''
