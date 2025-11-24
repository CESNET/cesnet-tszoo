from abc import ABC
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

import numpy as np

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
from cesnet_tszoo.utils.enums import TimeFormat


class TimeBasedHandler(ABC):
    def __init__(self,
                 logger: Logger,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 all_batch_size: int,
                 uses_all_time_period: bool,
                 sliding_window_size: int | None,
                 sliding_window_prediction_size: int | None,
                 sliding_window_step: int,
                 set_shared_size: float | int,
                 train_time_period: tuple[datetime, datetime] | range | float | None,
                 val_time_period: tuple[datetime, datetime] | range | float | None,
                 test_time_period: tuple[datetime, datetime] | range | float | None):

        self.train_time_period: Optional[np.ndarray] = train_time_period
        self.val_time_period: Optional[np.ndarray] = val_time_period
        self.test_time_period: Optional[np.ndarray] = test_time_period

        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.test_batch_size: int = test_batch_size
        self.all_batch_size: int = all_batch_size

        self.set_shared_size: int | float = set_shared_size

        self.sliding_window_size: Optional[int] = sliding_window_size
        self.sliding_window_prediction_size: Optional[int] = sliding_window_prediction_size
        self.sliding_window_step: int = sliding_window_step

        self.uses_all_time_period: bool = uses_all_time_period

        self.display_train_time_period: Optional[range] = None
        self.display_val_time_period: Optional[range] = None
        self.display_test_time_period: Optional[range] = None
        self.display_all_time_period: Optional[range] = None

        self.all_time_period: Optional[np.ndarray] = None

        self.logger: Logger = logger

    def _prepare_and_set_time_period_sets(self, all_time_ids: np.ndarray, time_format: TimeFormat) -> None:
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
        self.train_time_period, self.display_train_time_period = TimeBasedHandler._process_time_period(self.train_time_period, all_time_ids, self.logger, time_format, times_to_share)

        if self.train_time_period is not None:
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
        self.val_time_period, self.display_val_time_period = TimeBasedHandler._process_time_period(self.val_time_period, all_time_ids, self.logger, time_format, times_to_share)

        if self.val_time_period is not None:
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
        self.test_time_period, self.display_test_time_period = TimeBasedHandler._process_time_period(self.test_time_period, all_time_ids, self.logger, time_format, times_to_share)

        if self.test_time_period is not None:
            if self.sliding_window_size is not None and len(self.test_time_period) < self.sliding_window_size + self.sliding_window_prediction_size:
                raise ValueError("Sliding window size + prediction size is larger than the number of times in test_time_period.")
            self.logger.debug("Processed test_time_period: %s, display_test_time_period: %s", self.test_time_period, self.display_test_time_period)

        if self.uses_all_time_period:
            if self.train_time_period is None and self.val_time_period is None and self.test_time_period is None:
                self.all_time_period = all_time_ids.copy()
                self.all_time_period = TimeBasedHandler._set_time_period_form(self.all_time_period, all_time_ids, time_format, self.logger)
                self.logger.info("Using all times for all_time_period because train_time_period, val_time_period, and test_time_period are all set to None.")
            else:
                for temp_time_period in [self.train_time_period, self.val_time_period, self.test_time_period]:
                    if temp_time_period is None:
                        continue
                    elif self.all_time_period is None:
                        self.all_time_period = temp_time_period.copy()
                    else:
                        self.all_time_period = np.concatenate((self.all_time_period, temp_time_period))

                if self.train_time_period is not None:
                    self.logger.debug("all_time_period includes values from train_time_period.")
                if self.val_time_period is not None:
                    self.logger.debug("all_time_period includes values from val_time_period.")
                if self.test_time_period is not None:
                    self.logger.debug("all_time_period includes values from test_time_period.")

                self.all_time_period = np.unique(self.all_time_period)

        else:
            self.all_time_period = None

        if self.all_time_period is not None:
            self.display_all_time_period = range(self.all_time_period[ID_TIME_COLUMN_NAME][0], self.all_time_period[ID_TIME_COLUMN_NAME][-1] + 1)
            if self.sliding_window_size is not None and len(self.all_time_period) < self.sliding_window_size + self.sliding_window_prediction_size:
                raise ValueError("Sliding window size + prediction size is larger than the number of times in all_time_period.")
            self.logger.debug("Processed all_time_period: %s, display_all_time_period: %s", self.all_time_period, self.display_all_time_period)

    def _validate_time_periods_init(self):
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

    def _update_sliding_window(self, sliding_window_size: int | None, sliding_window_prediction_size: int | None, sliding_window_step: int | None, set_shared_size: float | int, all_time_ids: np.ndarray,
                               has_train: bool, has_val: bool, has_test: bool, has_all: bool):
        if isinstance(set_shared_size, float):
            assert set_shared_size >= 0 and set_shared_size <= 1, "set_shared_size float value must be between or equal to 0 and 1."
            set_shared_size = int(len(all_time_ids) * set_shared_size)

        assert set_shared_size >= 0, "set_shared_size must be of positive value."

        # Ensure sliding_window_size is either None or a valid integer greater than 1
        assert sliding_window_size is None or (isinstance(sliding_window_size, int) and sliding_window_size > 1), "sliding_window_size must be an integer greater than 1, or None."

        # Ensure sliding_window_prediction_size is either None or a valid integer greater or equal to 0
        assert sliding_window_prediction_size is None or (isinstance(sliding_window_prediction_size, int) and sliding_window_prediction_size >= 0), "sliding_window_prediction_size must be an integer greater than equal to 0, or None."

        # When sliding_window_prediction_size is set then sliding_window_size must be set too
        assert (sliding_window_size is None and sliding_window_prediction_size is None) or (sliding_window_size is not None), "When sliding_window_prediction_size is set then sliding_window_size must be set too."

        # Adjust batch sizes based on sliding_window_size
        if sliding_window_size is not None:

            if sliding_window_prediction_size is None:
                sliding_window_prediction_size = 0

            if sliding_window_step <= 0:
                raise ValueError("sliding_window_step must be greater or equal to 1.")

            total_window_size = sliding_window_size + sliding_window_prediction_size

            if set_shared_size == self.set_shared_size:
                if has_train and self.__get_time_count_in_period(self.train_time_period, len(all_time_ids)) < total_window_size:
                    raise ValueError("New sliding window size + prediction size is larger than the number of times in train_time_period.")

                if has_val and self.__get_time_count_in_period(self.val_time_period, len(all_time_ids)) < total_window_size:
                    raise ValueError("New sliding window size + prediction size is larger than the number of times in val_time_period.")

                if has_test and self.__get_time_count_in_period(self.test_time_period, len(all_time_ids)) < total_window_size:
                    raise ValueError("New sliding window size + prediction size is larger than the number of times in test_time_period.")

                if has_all and self.uses_all_time_period and self.__get_time_count_in_period(self.all_time_period, len(all_time_ids)) < total_window_size:
                    raise ValueError("New sliding window size + prediction size is larger than the number of times in all_time_period.")

            if total_window_size > self.train_batch_size:
                self.train_batch_size = sliding_window_size + sliding_window_prediction_size
                self.logger.info("train_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > self.val_batch_size:
                self.val_batch_size = sliding_window_size + sliding_window_prediction_size
                self.logger.info("val_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if total_window_size > self.test_batch_size:
                self.test_batch_size = sliding_window_size + sliding_window_prediction_size
                self.logger.info("test_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)
            if self.uses_all_time_period and total_window_size > self.all_batch_size:
                self.all_batch_size = sliding_window_size + sliding_window_prediction_size
                self.logger.info("all_batch_size adjusted to %s as it should be greater than or equal to sliding_window_size + sliding_window_prediction_size.", total_window_size)

        self.sliding_window_size = sliding_window_size
        self.sliding_window_prediction_size = sliding_window_prediction_size
        self.sliding_window_step = sliding_window_step
        self.set_shared_size = set_shared_size

    def _validate_set_shared_size_init(self):
        if isinstance(self.set_shared_size, float):
            assert self.set_shared_size >= 0 and self.set_shared_size <= 1, "set_shared_size float value must be between or equal to 0 and 1."

        assert self.set_shared_size >= 0, "set_shared_size must be of positive value."

    def __get_time_count_in_period(self, time_period: int | float | np.ndarray, all_time_ids_count: int) -> int:
        if isinstance(time_period, int):
            count = time_period
        elif isinstance(time_period, float):
            count = int(all_time_ids_count * time_period)
        else:
            count = len(time_period)

        return count

    def _validate_sliding_window_init(self):

        # Ensure sliding_window_size is either None or a valid integer greater than 1
        assert self.sliding_window_size is None or (isinstance(self.sliding_window_size, int) and self.sliding_window_size > 1), "sliding_window_size must be an integer greater than 1, or None."

        # Ensure sliding_window_prediction_size is either None or a valid integer greater or equal to 0
        assert self.sliding_window_prediction_size is None or (isinstance(self.sliding_window_prediction_size, int) and self.sliding_window_prediction_size >= 0), "sliding_window_prediction_size must be an integer greater than 0, or None."

        # When sliding_window_prediction_size is set then sliding_window_size must be set too
        assert (self.sliding_window_size is None and self.sliding_window_prediction_size is None) or (self.sliding_window_size is not None), "When sliding_window_prediction_size is set then sliding_window_size must be set too."

        if self.sliding_window_size is not None and self.sliding_window_prediction_size is None:
            self.sliding_window_prediction_size = 0

    def _validate_time_periods_overlap(self):
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

    @staticmethod
    def _process_time_period(time_period: np.ndarray, all_time_ids: np.ndarray, logger: Logger, time_format: TimeFormat, times_to_share: np.ndarray | None = None) -> np.ndarray | range:
        """Validates and filters the input `time_period` based on the `dataset` and `aggregation`. """

        if time_period is None:
            logger.debug("No time period provided, returning None for both time period and display time period.")
            return None, None

        elif isinstance(time_period, tuple):
            # Handle time period as a tuple of two datetime objects
            start_time = time_period[0].replace(tzinfo=timezone.utc).timestamp()
            end_time = time_period[1].replace(tzinfo=timezone.utc).timestamp()

            selected_time_mask = (all_time_ids[:][TIME_COLUMN_NAME] >= start_time) & (all_time_ids[:][TIME_COLUMN_NAME] < end_time)

            time_period = all_time_ids[selected_time_mask].copy()
            logger.debug("Selected time period based on start time %s and end time %s.", time_period[0], time_period[1])

        elif isinstance(time_period, range):
            # Handle time period as a range of indices
            indices, _ = zip(*all_time_ids)
            time_period = all_time_ids[np.where(np.isin(indices, [index for index in time_period]))].copy()
            logger.debug("Selected time period using indices from the provided range: %s", list(time_period))

        if times_to_share is not None:
            shareable_time_indices = np.where(np.isin(times_to_share[ID_TIME_COLUMN_NAME], time_period[ID_TIME_COLUMN_NAME], invert=True))[0]
            if len(shareable_time_indices) > 0:
                time_period = np.concatenate((times_to_share[shareable_time_indices], time_period))

        # Adjust time period to fit chosen time_format
        time_period = TimeBasedHandler._set_time_period_form(time_period, all_time_ids, time_format, logger)

        # Check if the time period ended up being empty after processing
        if len(time_period) == 0:
            logger.error("After processing, the time period ended up empty. Check the inputted time_periods for correctness.")
            raise ValueError("After processing time_period ended up empty. Check inputted time_periods.")

        # Display the selected time period range
        display_time_period = range(time_period[ID_TIME_COLUMN_NAME][0], time_period[ID_TIME_COLUMN_NAME][-1] + 1)
        logger.debug("Final time period selected: %s", display_time_period)

        return time_period, display_time_period

    @staticmethod
    def _set_time_period_form(time_period: np.ndarray, all_time_ids: np.ndarray, time_format: TimeFormat, logger: Logger) -> np.ndarray:
        """Sets the time period based on the selected `time_format`. """

        # Check the time format and process the time_period accordingly
        if time_format == TimeFormat.ID_TIME:
            temp = np.ndarray(time_period.shape, np.dtype([(ID_TIME_COLUMN_NAME, np.int32), (TIME_COLUMN_NAME, np.int32)]))
            temp[ID_TIME_COLUMN_NAME] = time_period[ID_TIME_COLUMN_NAME]
            temp[TIME_COLUMN_NAME] = time_period[TIME_COLUMN_NAME]
            logger.debug("Processed time_period using ID_TIME format.")

        elif time_format == TimeFormat.UNIX_TIME:
            temp = np.ndarray(time_period.shape, np.dtype([(ID_TIME_COLUMN_NAME, np.int32), (TIME_COLUMN_NAME, np.int32)]))
            temp[ID_TIME_COLUMN_NAME] = time_period[ID_TIME_COLUMN_NAME]
            temp[TIME_COLUMN_NAME] = time_period[TIME_COLUMN_NAME]
            logger.debug("Processed time_period using UNIX_TIME format.")

        elif time_format == TimeFormat.SHIFTED_UNIX_TIME:
            temp = np.ndarray(time_period.shape, np.dtype([(ID_TIME_COLUMN_NAME, np.int32), (TIME_COLUMN_NAME, np.int32)]))
            temp[ID_TIME_COLUMN_NAME] = time_period[ID_TIME_COLUMN_NAME]
            temp[TIME_COLUMN_NAME] = time_period[TIME_COLUMN_NAME] - all_time_ids[TIME_COLUMN_NAME][0]
            logger.debug("Processed time_period using SHIFTED_UNIX_TIME format with time shift applied.")

        elif time_format == TimeFormat.DATETIME:
            temp = np.ndarray(time_period.shape, np.dtype([(ID_TIME_COLUMN_NAME, np.int32), (TIME_COLUMN_NAME, datetime)]))
            temp[ID_TIME_COLUMN_NAME] = time_period[ID_TIME_COLUMN_NAME]

            for i in range(temp.shape[0]):
                temp[TIME_COLUMN_NAME][i] = datetime.fromtimestamp(time_period[TIME_COLUMN_NAME][i], tz=timezone.utc)
            logger.debug("Processed time_period using DATETIME format.")

        else:
            # This should not happen, raise an exception if an unsupported time_format is encountered
            logger.error("Unsupported time_format encountered: %s", time_format)
            raise ValueError("Invalid time_format specified. Should not happen.")

        logger.debug("Using '%s' time_format to set time_period.", time_format)

        return temp
