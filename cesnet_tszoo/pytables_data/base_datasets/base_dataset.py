import atexit
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import numpy.lib.recfunctions as rf
from torch.utils.data import Dataset
import torch

from cesnet_tszoo.pytables_data.utils.utils import load_database
from cesnet_tszoo.utils.enums import PreprocessType
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, ROW_START, ROW_END
from cesnet_tszoo.data_models.load_dataset_configs.load_config import LoadConfig
from cesnet_tszoo.data_models.holders import FillingHolder, TransformerHolder, AnomalyHandlerHolder, PerSeriesCustomHandlerHolder, AllSeriesCustomHandlerHolder, NoFitCustomHandlerHolder
from cesnet_tszoo.utils.enums import TimeFormat


class BaseDataset(Dataset, ABC):
    """Base class for PyTable wrappers. Used for main data loading... train, val, test etc."""

    def __init__(self, database_path: str, table_data_path: str, load_config: LoadConfig):
        self.load_config = deepcopy(load_config)
        self.saved_load_config = deepcopy(load_config)

        self.database_path = database_path
        self.table_data_path = table_data_path
        self.worker_id = None
        self.database = None
        self.table = None

        self.offset_exclude_feature_ids = len(self.load_config.features_to_take) - len(self.load_config.indices_of_features_to_take_no_ids)

        if self.load_config.include_time and self.load_config.time_format != TimeFormat.DATETIME:
            self.time_col_index = self.load_config.features_to_take.index(ID_TIME_COLUMN_NAME)

        if self.load_config.include_ts_id:
            self.ts_id_col_index = self.load_config.features_to_take.index(self.load_config.ts_id_name)
            self.ts_id_fill = self.load_config.ts_row_ranges[self.load_config.ts_id_name].reshape((self.load_config.ts_row_ranges.shape[0], 1))

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

    def load_data_from_table(self, ts_row_ranges_to_take: np.ndarray, time_indices_to_take: np.ndarray) -> np.ndarray:
        """Return data from table. Used preprocess will be applied. """

        result = np.full((len(ts_row_ranges_to_take), len(time_indices_to_take), len(self.load_config.features_to_take)), fill_value=np.nan, dtype=np.float64)

        for i, range_data in enumerate(ts_row_ranges_to_take):

            expected_offset = np.uint32(len(time_indices_to_take))
            real_offset = np.uint32(0)
            start = int(range_data[ROW_START])
            end = int(range_data[ROW_END])
            first_time_index = time_indices_to_take[0][ID_TIME_COLUMN_NAME]
            last_time_index = time_indices_to_take[-1][ID_TIME_COLUMN_NAME]

            # No more existing values
            if start >= end:
                result[i, :, self.offset_exclude_feature_ids:] = self._handle_data_preprocess(result[i, :, self.offset_exclude_feature_ids:].view(), i)

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

            if len(filtered_rows) > 0:
                result[i, existing_indices] = rf.structured_to_unstructured(filtered_rows[:][self.load_config.features_to_take], dtype=np.float64, copy=False)
                real_offset = len(filtered_rows)

            result[i, :, self.offset_exclude_feature_ids:] = self._handle_data_preprocess(result[i, :, self.offset_exclude_feature_ids:].view(), i)

            # Update ranges
            ts_row_ranges_to_take[ROW_START][i] = start + real_offset

        return result

    def _handle_data_preprocess(self, data: np.ndarray, idx: int) -> np.ndarray:
        for preprocess_note in self.load_config.preprocess_order:
            if preprocess_note.preprocess_type == PreprocessType.HANDLING_ANOMALIES:
                data = self._handle_anomalies(preprocess_note.holder, data, idx)
            elif preprocess_note.preprocess_type == PreprocessType.FILLING_GAPS:
                data = self._handle_filling(preprocess_note.holder, data, idx)
            elif preprocess_note.preprocess_type == PreprocessType.TRANSFORMING:
                data = self._handle_transforming(preprocess_note.holder, data, idx)
            elif preprocess_note.preprocess_type == PreprocessType.PER_SERIES_CUSTOM:
                data = self._handle_per_series_custom_handler(preprocess_note.holder, preprocess_note.can_be_applied, data, idx)
            elif preprocess_note.preprocess_type == PreprocessType.ALL_SERIES_CUSTOM:
                data = self._handle_all_series_custom_handler(preprocess_note.holder, preprocess_note.can_be_applied, data, idx)
            elif preprocess_note.preprocess_type == PreprocessType.NO_FIT_CUSTOM:
                data = self._handle_no_fit_custom_handler(preprocess_note.holder, preprocess_note.can_be_applied, data, idx)
            else:
                raise NotImplementedError()

        return data

    def _handle_filling(self, filling_holder: FillingHolder, data: np.ndarray, idx: int):
        """Fills data. """

        return filling_holder.apply(data, idx)

    def _handle_anomalies(self, anomaly_handler_holder: AnomalyHandlerHolder, data: np.ndarray, idx: int) -> np.ndarray:
        """Uses anomaly handlers. """

        if anomaly_handler_holder.is_empty():
            return data

        return anomaly_handler_holder.apply(data.view(), idx)

    def _handle_transforming(self, transfomer_holder: TransformerHolder, data: np.ndarray, idx: int) -> np.ndarray:
        """Uses transformers """

        if len(self.load_config.indices_of_features_to_take_no_ids) == 1:
            data = transfomer_holder.apply(data.reshape(-1, 1), idx)
        elif len(self.load_config.time_period) == 1:
            data = transfomer_holder.apply(data.reshape(1, -1), idx)
        else:
            data = transfomer_holder.apply(data, idx)

        return data

    def _handle_per_series_custom_handler(self, handler_holder: PerSeriesCustomHandlerHolder, can_apply: bool, data: np.ndarray, idx: int) -> np.ndarray:
        if can_apply:
            data = handler_holder.apply(data, idx)

        return data

    def _handle_all_series_custom_handler(self, handler_holder: AllSeriesCustomHandlerHolder, can_apply: bool, data: np.ndarray, idx: int) -> np.ndarray:
        if can_apply:
            data = handler_holder.apply(data, idx)

        return data

    def _handle_no_fit_custom_handler(self, handler_holder: NoFitCustomHandlerHolder, can_apply: bool, data: np.ndarray, idx: int) -> np.ndarray:
        if can_apply:
            data = handler_holder.apply(data, idx)

        return data

    @staticmethod
    def worker_init_fn(worker_id) -> None:
        """Inits dataset instance used by worker. """

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.pytables_worker_init(worker_id)
