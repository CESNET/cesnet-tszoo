import atexit
from abc import ABC, abstractmethod
from copy import deepcopy

from torch.utils.data import Dataset
import torch
import numpy as np
import numpy.lib.recfunctions as rf

from cesnet_tszoo.utils.enums import PreprocessType
from cesnet_tszoo.data_models.init_dataset_configs.init_config import DatasetInitConfig
from cesnet_tszoo.data_models.preprocess_order_group import PreprocessOrderGroup
from cesnet_tszoo.data_models.holders import FillingHolder, TransformerHolder, AnomalyHandlerHolder, PerSeriesCustomHandlerHolder, AllSeriesCustomHandlerHolder, NoFitCustomHandlerHolder
from cesnet_tszoo.utils.constants import ROW_START, ROW_END, ID_TIME_COLUMN_NAME
from cesnet_tszoo.pytables_data.utils.utils import load_database


class InitializerDataset(Dataset, ABC):
    """Base class for initializer PyTable wrappers. Used for going through data to fit transformers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, init_config: DatasetInitConfig):
        self.database_path = database_path
        self.table_data_path = table_data_path
        self.init_config = deepcopy(init_config)
        self.table = None
        self.worker_id = None
        self.database = None

        self.offset_exclude_feature_ids = len(self.init_config.features_to_take) - len(self.init_config.indices_of_features_to_take_no_ids)

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

    def load_data_from_table(self, identifier_row_range_to_take: np.ndarray) -> np.ndarray:
        """Returns data from the table and indices of rows where values exists."""

        result = np.full((len(self.init_config.time_period), len(self.init_config.features_to_take)), fill_value=np.nan, dtype=np.float64)
        result[:, self.offset_exclude_feature_ids:] = np.nan

        expected_offset = np.uint32(len(self.init_config.time_period))
        start = int(identifier_row_range_to_take[ROW_START])
        end = int(identifier_row_range_to_take[ROW_END])
        first_time_index = self.init_config.time_period[0][ID_TIME_COLUMN_NAME]
        last_time_index = self.init_config.time_period[-1][ID_TIME_COLUMN_NAME]

        # No more existing values
        if start >= end:
            return result, np.array([], dtype=int)

        # Expected range for times in time series
        if expected_offset + start >= end:
            expected_offset = end - start

        # Naive getting data from table
        rows = self.table[start: start + expected_offset]

        # Getting more data if needed... if received data could contain more relevant data
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

        # missing_values_mask = np.ones(len(self.init_config.time_period), dtype=bool)
        # missing_values_mask[existing_indices] = 0
        # xmissing_indices = np.nonzero(missing_values_mask)[0]

        if len(filtered_rows) > 0:
            result[existing_indices, :] = rf.structured_to_unstructured(filtered_rows[:][self.init_config.features_to_take], dtype=np.float64, copy=False)

        # result, count_values = self._handle_data_preprocess(result, missing_indices, missing_values_mask, idx)

        return result, existing_indices

    @abstractmethod
    def _handle_data_preprocess(self, data: np.ndarray, idx: int) -> np.ndarray:
        ...

    def _handle_data_preprocess_order_group(self, preprocess_order_group: PreprocessOrderGroup, data: np.ndarray, idx: int) -> np.ndarray:
        for preprocess_order in preprocess_order_group.preprocess_inner_orders:
            if preprocess_order.preprocess_type == PreprocessType.HANDLING_ANOMALIES:
                self._handle_anomalies(preprocess_order.holder, preprocess_order.should_be_fitted, data, idx)
            elif preprocess_order.preprocess_type == PreprocessType.FILLING_GAPS:
                self._handle_filling(preprocess_order.holder, data, idx)
            elif preprocess_order.preprocess_type == PreprocessType.TRANSFORMING:
                data = self._handle_transforming(preprocess_order.holder, preprocess_order.should_be_fitted, data, idx)
            elif preprocess_order.preprocess_type == PreprocessType.PER_SERIES_CUSTOM:
                data = self._handle_per_series_custom_handler(preprocess_order.holder, preprocess_order.should_be_fitted, preprocess_order.can_be_applied, data, idx)
            elif preprocess_order.preprocess_type == PreprocessType.ALL_SERIES_CUSTOM:
                data = self._handle_all_series_custom_handler(preprocess_order.holder, preprocess_order.should_be_fitted, preprocess_order.can_be_applied, data, idx)
            elif preprocess_order.preprocess_type == PreprocessType.NO_FIT_CUSTOM:
                data = self._handle_no_fit_custom_handler(preprocess_order.holder, preprocess_order.can_be_applied, data, idx)
            else:
                raise NotImplementedError()

        return data

    @abstractmethod
    def _handle_filling(self, filling_holder: FillingHolder, data: np.ndarray, idx: int):
        """Fills data. """
        ...

    @abstractmethod
    def _handle_anomalies(self, anomaly_handler_holder: AnomalyHandlerHolder, should_fit: bool, data: np.ndarray, idx: int):
        """Fits and uses anomaly handlers. """
        ...

    @abstractmethod
    def _handle_transforming(self, transfomer_holder: TransformerHolder, should_fit: bool, data: np.ndarray, idx: int) -> np.ndarray:
        """Fits and uses transformers """
        ...

    def _handle_per_series_custom_handler(self, handler_holder: PerSeriesCustomHandlerHolder, should_fit: bool, can_apply: bool, data: np.ndarray, idx: int) -> np.ndarray:

        if should_fit:
            handler_holder.get_instance(idx).fit(data)

        if can_apply:
            data = handler_holder.get_instance(idx).apply(data)

        return data

    def _handle_all_series_custom_handler(self, handler_holder: AllSeriesCustomHandlerHolder, should_fit: bool, can_apply: bool, data: np.ndarray, idx: int) -> np.ndarray:
        if should_fit:
            handler_holder.get_instance(idx).partial_fit(data)

        if can_apply:
            data = handler_holder.get_instance(idx).apply(data)

        return data

    def _handle_no_fit_custom_handler(self, handler_holder: NoFitCustomHandlerHolder, can_apply: bool, data: np.ndarray, idx: int) -> np.ndarray:
        if can_apply:
            data = handler_holder.get_instance(idx).apply(data)

        return data

    @staticmethod
    def worker_init_fn(worker_id) -> None:
        """Inits dataset instace used by worker. """

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.pytables_worker_init(worker_id)
