import atexit
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.lib.recfunctions as rf
from torch.utils.data import Dataset
import torch

from cesnet_tszoo.pytables_data.utils.utils import load_database
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, ROW_START, ROW_END
from cesnet_tszoo.utils.filler import Filler
from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.utils.anomaly_handler import AnomalyHandler
from cesnet_tszoo.utils.enums import TimeFormat


class BaseDataset(Dataset, ABC):
    """Base class for PyTable wrappers. Used for main data loading... train, val, test etc."""

    def __init__(self, database_path: str, table_data_path: str, ts_id_name: str, ts_row_ranges: np.ndarray, time_period: np.ndarray, features_to_take: list[str], indices_of_features_to_take_no_ids: list[int],
                 default_values: np.ndarray, fillers: np.ndarray[Filler], include_time: bool, include_ts_id: bool, time_format: TimeFormat, transformers: np.ndarray[Transformer] | Transformer | None,
                 anomaly_handlers: np.ndarray[AnomalyHandler]):

        self.database_path = database_path
        self.table_data_path = table_data_path
        self.ts_id_name = ts_id_name
        self.worker_id = None
        self.database = None
        self.table = None

        self.time_period = time_period
        self.features_to_take = features_to_take
        self.indices_of_features_to_take_no_ids = indices_of_features_to_take_no_ids
        self.default_values = default_values
        self.transformers = transformers
        self.anomaly_handlers = anomaly_handlers
        self.offset_exclude_feature_ids = len(self.features_to_take) - len(self.indices_of_features_to_take_no_ids)

        self.fillers = deepcopy(fillers)

        self.ts_row_ranges = deepcopy(ts_row_ranges)

        self.time_format = time_format
        self.include_time = include_time
        if self.include_time and self.time_format != TimeFormat.DATETIME:
            self.time_col_index = self.features_to_take.index(ID_TIME_COLUMN_NAME)

        self.include_ts_id = include_ts_id
        if self.include_ts_id:
            self.ts_id_col_index = self.features_to_take.index(self.ts_id_name)
            self.ts_id_fill = self.ts_row_ranges[self.ts_id_name].reshape((self.ts_row_ranges.shape[0], 1))

    @abstractmethod
    def __getitem__(self, index):
        ...

    @abstractmethod
    def __len__(self):
        ...

    def pytables_worker_init(self, worker_id: int = 0) -> None:
        """Prepare this dataset for loading data. """

        self.worker_id = worker_id

        self.database, self.table = load_database(dataset_path=self.database_path, table_data_path=self.table_data_path)
        atexit.register(self.cleanup)

    def cleanup(self) -> None:
        """Clean used resources. """

        self.database.close()

    def load_data_from_table(self, ts_row_ranges_to_take: np.ndarray, time_indices_to_take: np.ndarray,
                             fillers_to_use: Optional[np.ndarray[Filler]], anomaly_handlers_to_use: Optional[np.ndarray[AnomalyHandler]]) -> np.ndarray:
        """Return data from table. Missing values are filled with `fillers_to_use` and `default_values`. Anomalies are handled by `anomaly_handlers_to_use`. """

        result = np.full((len(ts_row_ranges_to_take), len(time_indices_to_take), len(self.features_to_take)), fill_value=np.nan, dtype=np.float64)

        full_missing_indices = np.arange(0, len(time_indices_to_take))

        for i, range_data in enumerate(ts_row_ranges_to_take):

            expected_offset = np.uint32(len(time_indices_to_take))
            real_offset = np.uint32(0)
            start = int(range_data[ROW_START])
            end = int(range_data[ROW_END])
            first_time_index = time_indices_to_take[0][ID_TIME_COLUMN_NAME]
            last_time_index = time_indices_to_take[-1][ID_TIME_COLUMN_NAME]

            # No more existing values
            if start >= end:
                result[i, :, self.offset_exclude_feature_ids:] = self.default_values

                fillers_to_use[i].fill(result[i, :, self.offset_exclude_feature_ids:].view(), np.array([]), full_missing_indices, default_values=self.default_values,
                                       first_next_existing_values=None, first_next_existing_values_distance=None)
                continue

            # Expected range for times in time series
            if expected_offset + start >= end:
                expected_offset = end - start

            # Naive getting data from table
            rows = self.table[start: start + expected_offset]

            # Getting more date if needed... if received data could contain more relevant data
            if rows[-1][ID_TIME_COLUMN_NAME] < first_time_index:
                if start + expected_offset + last_time_index - rows[-1][ID_TIME_COLUMN_NAME] >= end:
                    rows = self.table[start + expected_offset: end]
                else:
                    rows = self.table[start + expected_offset: start + expected_offset + last_time_index - rows[-1][ID_TIME_COLUMN_NAME]]
            elif rows[-1][ID_TIME_COLUMN_NAME] < last_time_index:
                if start + expected_offset + last_time_index - rows[-1][ID_TIME_COLUMN_NAME] >= end:
                    rows = self.table[start: end]
                else:
                    rows = self.table[start: start + expected_offset + last_time_index - rows[-1][ID_TIME_COLUMN_NAME]]

            rows_id_times = rows[:][ID_TIME_COLUMN_NAME]
            upper_mask = (rows_id_times <= last_time_index)
            lower_mask = (rows_id_times >= first_time_index)
            mask = lower_mask & upper_mask

            # Get valid times
            filtered_rows = rows[mask].view()
            filtered_rows[ID_TIME_COLUMN_NAME] = filtered_rows[ID_TIME_COLUMN_NAME] - first_time_index
            existing_indices = filtered_rows[ID_TIME_COLUMN_NAME].view()

            missing_values_mask = np.ones(len(time_indices_to_take), dtype=bool)
            missing_values_mask[existing_indices] = 0
            missing_indices = np.nonzero(missing_values_mask)[0]

            if len(filtered_rows) > 0:
                result[i, existing_indices] = rf.structured_to_unstructured(filtered_rows[:][self.features_to_take], dtype=np.float64, copy=False)
                real_offset = len(filtered_rows)

            first_next_existing_values = None
            first_next_existing_values_distance = None

            if len(existing_indices) != len(time_indices_to_take):
                upper_valid_rows = rows[lower_mask].view()
                if len(upper_valid_rows) != len(existing_indices) and upper_valid_rows[ID_TIME_COLUMN_NAME][len(existing_indices)] <= self.time_period[ID_TIME_COLUMN_NAME][-1]:
                    first_next_existing_values = rf.structured_to_unstructured(upper_valid_rows[len(existing_indices)][self.features_to_take], dtype=np.float64, copy=False)
                    first_next_existing_values_distance = upper_valid_rows[ID_TIME_COLUMN_NAME][len(existing_indices)]

            if anomaly_handlers_to_use is not None:
                anomaly_handlers_to_use[i].transform_anomalies(result[i, :, self.offset_exclude_feature_ids:].view())

            result[i, missing_indices, self.offset_exclude_feature_ids:] = self.default_values

            fillers_to_use[i].fill(result[i, :, self.offset_exclude_feature_ids:].view(), existing_indices, missing_indices, default_values=self.default_values,
                                   first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

            # Update ranges
            ts_row_ranges_to_take[ROW_START][i] = start + real_offset

        return result

    @staticmethod
    def worker_init_fn(worker_id) -> None:
        """Inits dataset instance used by worker. """

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.pytables_worker_init(worker_id)
