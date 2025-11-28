from abc import ABC
from logging import Logger
from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from cesnet_tszoo.utils.constants import ROW_END, ROW_START


class SeriesBasedHandler(ABC):
    def __init__(self,
                 logger: Logger,
                 uses_all_ts: bool,
                 train_ts: list[int] | npt.NDArray[np.int_] | float | int | None,
                 val_ts: list[int] | npt.NDArray[np.int_] | float | int | None,
                 test_ts: list[int] | npt.NDArray[np.int_] | float | int | None):

        self.train_ts: Optional[np.ndarray] = train_ts
        self.val_ts: Optional[np.ndarray] = val_ts
        self.test_ts: Optional[np.ndarray] = test_ts
        self.all_ts: Optional[np.ndarray] = None

        self.uses_all_ts: bool = uses_all_ts

        self.train_ts_row_ranges: Optional[np.ndarray] = None
        self.val_ts_row_ranges: Optional[np.ndarray] = None
        self.test_ts_row_ranges: Optional[np.ndarray] = None
        self.all_ts_row_ranges: Optional[np.ndarray] = None

        self.logger: Logger = logger

    def _prepare_and_set_ts_sets(self, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray, ts_id_name: str, random_state: Optional[int], rd: np.random.RandomState) -> None:
        """Validates and filters the input time series IDs based on the `dataset` and `source_type`. Handles random split."""

        random_ts_ids = all_ts_ids[ts_id_name]
        random_indices = np.arange(len(all_ts_ids))

        # Process train_ts if it was specified with times series ids
        if self.train_ts is not None and not isinstance(self.train_ts, (float, int)):
            self.train_ts, self.train_ts_row_ranges, _ = SeriesBasedHandler._process_ts_ids(self.train_ts, all_ts_ids, all_ts_row_ranges, None, None, self.logger, ts_id_name, random_state, rd)

            mask = np.isin(random_ts_ids, self.train_ts, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("train_ts set: %s", self.train_ts)

        # Process val_ts if it was specified with times series ids
        if self.val_ts is not None and not isinstance(self.val_ts, (float, int)):
            self.val_ts, self.val_ts_row_ranges, _ = SeriesBasedHandler._process_ts_ids(self.val_ts, all_ts_ids, all_ts_row_ranges, None, None, self.logger, ts_id_name, random_state, rd)

            mask = np.isin(random_ts_ids, self.val_ts, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("val_ts set: %s", self.val_ts)

        # Process time_ts if it was specified with times series ids
        if self.test_ts is not None and not isinstance(self.test_ts, (float, int)):
            self.test_ts, self.test_ts_row_ranges, _ = SeriesBasedHandler._process_ts_ids(self.test_ts, all_ts_ids, all_ts_row_ranges, None, None, self.logger, ts_id_name, random_state, rd)

            mask = np.isin(random_ts_ids, self.test_ts, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("test_ts set: %s", self.test_ts)

        # Convert proportions to total values
        if isinstance(self.train_ts, float):
            self.train_ts = int(self.train_ts * len(random_ts_ids))
            self.logger.debug("train_ts converted to total values: %s", self.train_ts)
        if isinstance(self.val_ts, float):
            self.val_ts = int(self.val_ts * len(random_ts_ids))
            self.logger.debug("val_ts converted to total values: %s", self.val_ts)
        if isinstance(self.test_ts, float):
            self.test_ts = int(self.test_ts * len(random_ts_ids))
            self.logger.debug("test_ts converted to total values: %s", self.test_ts)

        # Process random train_ts if it is to be randomly made
        if isinstance(self.train_ts, int):
            self.train_ts, self.train_ts_row_ranges, random_indices = SeriesBasedHandler._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.train_ts, random_indices, self.logger, ts_id_name, random_state, rd)
            self.logger.debug("Random train_ts set with %s time series.", self.train_ts)

        # Process random val_ts if it is to be randomly made
        if isinstance(self.val_ts, int):
            self.val_ts, self.val_ts_row_ranges, random_indices = SeriesBasedHandler._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.val_ts, random_indices, self.logger, ts_id_name, random_state, rd)
            self.logger.debug("Random val_ts set with %s time series.", self.val_ts)

        # Process random test_ts if it is to be randomly made
        if isinstance(self.test_ts, int):
            self.test_ts, self.test_ts_row_ranges, random_indices = SeriesBasedHandler._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.test_ts, random_indices, self.logger, ts_id_name, random_state, rd)
            self.logger.debug("Random test_ts set with %s time series.", self.test_ts)

        if self.uses_all_ts:
            if self.train_ts is None and self.val_ts is None and self.test_ts is None:
                self.all_ts = all_ts_ids[ts_id_name]
                self.all_ts, self.all_ts_row_ranges, _ = SeriesBasedHandler._process_ts_ids(self.all_ts, all_ts_ids, all_ts_row_ranges, None, None, self.logger, ts_id_name, random_state, rd)
                self.logger.info("Using all time series for all_ts because train_ts, val_ts, and test_ts are all set to None.")
            else:
                for temp_ts_ids in [self.train_ts, self.val_ts, self.test_ts]:
                    if temp_ts_ids is None:
                        continue
                    elif self.all_ts is None:
                        self.all_ts = temp_ts_ids.copy()
                    else:
                        self.all_ts = np.concatenate((self.all_ts, temp_ts_ids))

                if self.train_ts is not None:
                    self.logger.debug("all_ts includes ids from train_ts.")
                if self.val_ts is not None:
                    self.logger.debug("all_ts includes ids from val_ts.")
                if self.test_ts is not None:
                    self.logger.debug("all_ts includes ids from test_ts.")

                self.all_ts, self.all_ts_row_ranges, _ = self._process_ts_ids(self.all_ts, all_ts_ids, all_ts_row_ranges, None, None, self.logger, ts_id_name, random_state, rd)
        else:
            self.all_ts = None

        if self.all_ts is not None:
            self.logger.debug("all_ts set with %s time series.", self.all_ts)

    def _validate_ts_init(self):
        split_float_total = 0

        if isinstance(self.train_ts, (float, int)):
            assert self.train_ts > 0, "train_ts must be greater than 0."
            if isinstance(self.train_ts, float):
                split_float_total += self.train_ts

        if isinstance(self.val_ts, (float, int)):
            assert self.val_ts > 0, "val_ts must be greater than 0"
            if isinstance(self.val_ts, float):
                split_float_total += self.val_ts

        if isinstance(self.test_ts, (float, int)):
            assert self.test_ts > 0, "test_ts must be greater than 0"
            if isinstance(self.test_ts, float):
                split_float_total += self.test_ts

        # Check if the total of float splits exceeds 1.0
        if split_float_total > 1.0:
            self.logger.error("The total of the float split sizes is greater than 1.0. Current total: %s", split_float_total)
            raise ValueError("Total value of used float split sizes can't be larger than 1.0.")

    def _validate_ts_overlap(self):
        train_size = 0
        if self.train_ts is not None:
            train_size += len(self.train_ts)

        val_size = 0
        if self.val_ts is not None:
            val_size += len(self.val_ts)

        test_size = 0
        if self.test_ts is not None:
            test_size += len(self.test_ts)

        # Check for overlap between train, val, and test sets
        if train_size + val_size + test_size > 0 and train_size + val_size + test_size != len(np.unique(self.all_ts)):
            self.logger.error("Overlap detected! Train, Val, and Test sets can't have the same IDs.")
            raise ValueError("Train, Val, and Test can't have the same IDs.")

    @staticmethod
    def _process_ts_ids(ts_ids: np.ndarray, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray, split_size: float | int | None, random_indices: np.ndarray, logger: Logger, ts_id_name: str, random_state: Optional[int], rd: np.random.RandomState) -> None:
        """Validates and filters the input `ts_ids` based on the `dataset` and `source_type`. """

        if ts_ids is None and split_size is None:
            logger.debug("Both ts_ids and split_size are None, returning None for ts_ids and ts_row_ranges.")
            return None, None, random_indices

        if split_size is not None:
            if split_size > len(random_indices):
                raise ValueError(f"Trying to use more time series than there are in the dataset. There are {len(all_ts_ids)} time series available.")

            if split_size == len(random_indices):
                rd.shuffle(random_indices)
                ts_indices = random_indices
                ts_ids = all_ts_ids[ts_id_name][ts_indices]
                random_indices = np.array([])  # No remaining indices
                logger.debug("Using all random indices. Shuffling complete, no remaining indices.")
            else:
                ts_indices, random_indices = train_test_split(random_indices, train_size=split_size, random_state=random_state)
                ts_ids = all_ts_ids[ts_id_name][ts_indices]
                logger.debug("Split random indices into train (size=%s) and remaining.", split_size)
        else:
            # Handling for the case where split_size is None, using provided ts_ids directly
            ts_ids = np.array(ts_ids)
            temp = ts_ids

            _, idx = np.unique(ts_ids, True, False, False)
            idx = np.sort(idx)
            ts_ids = ts_ids[idx]

            ts_indices = [np.where(all_ts_ids[ts_id_name] == x)[0][0] for x in ts_ids]
            ts_ids = all_ts_ids[ts_id_name][ts_indices]

            if len(ts_ids) == 0:
                logger.error("After processing, ts_ids ended up empty. Check the inputted ts_ids for correctness.")
                raise ValueError("After processing, ts_ids ended up empty. Check the inputted ts_ids.")

            if len(temp) != len(ts_ids):
                logger.warning("Some invalid Time Series IDs were removed from ts_ids. Adjusting to only valid ts_ids.")

        # Process the row ranges for the selected time series indices
        temp = all_ts_row_ranges[ts_indices]
        ts_row_ranges = np.ndarray(len(temp), dtype=[(ts_id_name, np.uint32), (ROW_START, np.uint64), (ROW_END, np.uint64)])
        ts_row_ranges[ts_id_name] = temp[ts_id_name]
        ts_row_ranges[ROW_START] = temp[ROW_START]
        ts_row_ranges[ROW_END] = temp[ROW_END]

        logger.debug("Returning ts_ids and ts_row_ranges for selected time series.")

        return ts_ids, ts_row_ranges, random_indices
