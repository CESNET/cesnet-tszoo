from copy import deepcopy

import numpy as np

from cesnet_tszoo.utils.filler import Filler
from cesnet_tszoo.utils.anomaly_handler import AnomalyHandler
from cesnet_tszoo.pytables_data.base_datasets.initializer_dataset import InitializerDataset


class DisjointTimeBasedInitializerDataset(InitializerDataset):
    """Used for time based datasets. Used for going through data to fit transformers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, ts_id_name: str, ts_row_ranges: np.ndarray, time_period: np.ndarray,
                 features_to_take: list[str], indices_of_features_to_take_no_ids: list[int], default_values: np.ndarray, fillers: np.ndarray[Filler],
                 anomaly_handlers: np.ndarray[AnomalyHandler]):

        self.ts_row_ranges = ts_row_ranges
        self.time_period = time_period

        self.anomaly_handlers = anomaly_handlers

        self.fillers = deepcopy(fillers)

        super(DisjointTimeBasedInitializerDataset, self).__init__(database_path, table_data_path, ts_id_name, time_period, features_to_take, indices_of_features_to_take_no_ids, default_values)

    def __getitem__(self, idx):

        data, count_values = self.load_data_from_table(self.ts_row_ranges[idx], idx)
        this_anomaly_handler = self.anomaly_handlers[idx] if self.anomaly_handlers is not None else None

        # Prepare data from current time series for training transformer if needed
        if len(self.indices_of_features_to_take_no_ids) == 1:
            train_data = data[: len(self.time_period), self.offset_exclude_feature_ids:].reshape(-1, 1)
        elif len(self.time_period) == 1:
            train_data = data[: len(self.time_period), self.offset_exclude_feature_ids:].reshape(1, -1)
        else:
            train_data = data[: len(self.time_period), self.offset_exclude_feature_ids:]

        return train_data, count_values, this_anomaly_handler

    def __len__(self) -> int:
        return len(self.ts_row_ranges)

    def fill_values(self, missing_values_mask: np.ndarray, idx, result, first_next_existing_values, first_next_existing_values_distance):
        """Just fills data. """

        existing_indices = np.where(missing_values_mask == 0)[0]
        missing_indices = np.where(missing_values_mask == 1)[0]

        if self.fillers is not None:
            self.fillers[idx].fill(result[:, self.offset_exclude_feature_ids:].view(), existing_indices, missing_indices, default_values=self.default_values,
                                   first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

        return (len(existing_indices), len(missing_indices))

    def handle_anomalies(self, data: np.ndarray, idx: int):
        """Fits and uses anomaly handlers. """

        if self.anomaly_handlers is None:
            return

        self.anomaly_handlers[idx].fit(data[:, self.offset_exclude_feature_ids:])
        self.anomaly_handlers[idx].transform_anomalies(data[:, self.offset_exclude_feature_ids:], self.default_values)
