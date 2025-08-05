import atexit
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
import torch
import numpy as np
import numpy.lib.recfunctions as rf

from cesnet_tszoo.utils.constants import ROW_START, ROW_END, ID_TIME_COLUMN_NAME
from cesnet_tszoo.pytables_data.utils.utils import load_database


class InitializerDataset(Dataset, ABC):
    """Base class for initializer PyTable wrappers. Used for going through data to fit transformers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, ts_id_name: str, time_period: np.ndarray, features_to_take: list[str], indices_of_features_to_take_no_ids: list[int], default_values: np.ndarray):
        self.database_path = database_path
        self.table_data_path = table_data_path
        self.ts_id_name = ts_id_name
        self.table = None
        self.worker_id = None
        self.database = None

        self.features_to_take = features_to_take
        self.indices_of_features_to_take_no_ids = indices_of_features_to_take_no_ids

        self.default_values = default_values

        self.offset_exclude_feature_ids = len(self.features_to_take) - len(self.indices_of_features_to_take_no_ids)

        self.time_period = time_period

    def pytables_worker_init(self, worker_id=0) -> None:
        """Prepares this dataset for loading data. """

        self.worker_id = worker_id

        self.database, self.table = load_database(dataset_path=self.database_path, table_data_path=self.table_data_path)
        atexit.register(self.cleanup)

    @abstractmethod
    def __getitem__(self, index):
        ...

    @abstractmethod
    def __len__(self):
        ...

    def cleanup(self) -> None:
        """Cleans used resources. """

        self.database.close()

    def load_data_from_table(self, identifier_row_range_to_take: np.ndarray, idx: int) -> np.ndarray:
        """Returns data from table. Missing values are filled fillers and `default_values`. Prepares fillers."""

        result = np.full((len(self.time_period), len(self.features_to_take)), fill_value=np.nan, dtype=np.float64)
        result[:, self.offset_exclude_feature_ids:] = self.default_values

        expected_offset = np.uint32(len(self.time_period))
        start = int(identifier_row_range_to_take[ROW_START])
        end = int(identifier_row_range_to_take[ROW_END])
        first_time_index = self.time_period[0][ID_TIME_COLUMN_NAME]
        last_time_index = self.time_period[-1][ID_TIME_COLUMN_NAME]

        # No more existing values
        if start >= end:
            missing_values_mask = np.ones(len(self.time_period), dtype=bool)

            count_values = self.fill_values(missing_values_mask, idx, result, None, None)

            return result, count_values

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
                rows = self.table[start + expected_offset: start + expected_offset + last_time_index - rows[len(rows) - 1][ID_TIME_COLUMN_NAME]]
        elif rows[-1][ID_TIME_COLUMN_NAME] < last_time_index:
            if start + expected_offset + last_time_index - rows[len(rows) - 1][ID_TIME_COLUMN_NAME] >= end:
                rows = self.table[start: end]
            else:
                rows = self.table[start: start + expected_offset + last_time_index - rows[len(rows) - 1][ID_TIME_COLUMN_NAME]]

        rows_id_times = rows[:][ID_TIME_COLUMN_NAME]
        mask = (rows_id_times >= first_time_index) & (rows_id_times <= last_time_index)

        # Get valid times
        filtered_rows = rows[mask].view()
        filtered_rows[ID_TIME_COLUMN_NAME] = filtered_rows[ID_TIME_COLUMN_NAME] - first_time_index
        existing_indices = filtered_rows[ID_TIME_COLUMN_NAME].view()

        missing_values_mask = np.ones(len(self.time_period), dtype=bool)
        missing_values_mask[existing_indices] = 0

        if len(filtered_rows) > 0:
            result[existing_indices, :] = rf.structured_to_unstructured(filtered_rows[:][self.features_to_take], dtype=np.float64, copy=False)

        count_values = self.fill_values(missing_values_mask, idx, result, None, None)

        return result, count_values

    @abstractmethod
    def fill_values(self, missing_values_mask: np.ndarray, idx, result, first_next_existing_values, first_next_existing_values_distance):
        """Fills data. """

        ...

    @staticmethod
    def worker_init_fn(worker_id) -> None:
        """Inits dataset instace used by worker. """

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.pytables_worker_init(worker_id)
