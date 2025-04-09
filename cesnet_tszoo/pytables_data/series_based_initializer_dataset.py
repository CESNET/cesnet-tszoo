from copy import deepcopy

import numpy as np

from cesnet_tszoo.utils.filler import Filler
from cesnet_tszoo.pytables_data.base_datasets.initializer_dataset import InitializerDataset


class SeriesBasedInitializerDataset(InitializerDataset):
    """Used for series based datasets. Used for going through data to fit scalers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, ts_id_name: str, train_ts_row_ranges: np.ndarray, val_ts_row_ranges: np.ndarray, test_ts_row_ranges: np.ndarray, all_ts_row_ranges: np.ndarray, time_period: np.ndarray,
                 features_to_take: list[str], indices_of_features_to_take_no_ids: list[int], default_values: np.ndarray, all_fillers: np.ndarray[Filler]):
        self.train_ts_row_ranges = train_ts_row_ranges
        self.val_ts_row_ranges = val_ts_row_ranges
        self.test_ts_row_ranges = test_ts_row_ranges
        self.all_ts_row_ranges = all_ts_row_ranges

        self.all_fillers = deepcopy(all_fillers)

        super(SeriesBasedInitializerDataset, self).__init__(database_path, table_data_path, ts_id_name, time_period, features_to_take, indices_of_features_to_take_no_ids, default_values)

    def __getitem__(self, idx):

        current_ts_row_ranges = None
        offset = 0
        is_train = False
        is_val = False
        is_test = False

        # Check if current time series belongs into train set
        if self.train_ts_row_ranges is not None:
            if idx < len(self.train_ts_row_ranges):
                current_ts_row_ranges = self.train_ts_row_ranges
                is_train = True
            else:
                offset += len(self.train_ts_row_ranges)

        if self.val_ts_row_ranges is not None and not is_train:
            if idx - offset >= 0 and idx - offset < len(self.val_ts_row_ranges):
                current_ts_row_ranges = self.val_ts_row_ranges
                is_val = True
            elif idx - offset > 0:
                offset += len(self.val_ts_row_ranges)

        if self.test_ts_row_ranges is not None and not is_train and not is_val:
            if idx - offset >= 0 and idx - offset < len(self.test_ts_row_ranges):
                current_ts_row_ranges = self.test_ts_row_ranges
                is_test = True

        if not is_train and not is_val and not is_test:
            current_ts_row_ranges = self.all_ts_row_ranges

        data, count_values = self.load_data_from_table(current_ts_row_ranges[idx - offset], idx)

        # If current times series belongs to train set, return it for fitting scalers
        if is_train:
            train_data = None

            if len(self.indices_of_features_to_take_no_ids) == 1:
                train_data = data[:, self.offset_exclude_feature_ids:].reshape(-1, 1)
            elif len(self.time_period) == 1:
                train_data = data[:, self.offset_exclude_feature_ids:].reshape(1, -1)
            else:
                train_data = data[:, self.offset_exclude_feature_ids:]

            return train_data, count_values, is_train, is_val, is_test, idx - offset

        return None, count_values, is_train, is_val, is_test, idx - offset

    def __len__(self) -> int:
        return len(self.all_ts_row_ranges)

    def fill_values(self, missing_values_mask: np.ndarray, idx, result, first_next_existing_values, first_next_existing_values_distance):
        """Just fills data. """

        existing_indices = np.where(missing_values_mask == 0)[0]
        missing_indices = np.where(missing_values_mask == 1)[0]

        if self.all_fillers is not None:
            self.all_fillers[idx].fill(result[:, self.offset_exclude_feature_ids:].view(), existing_indices, missing_indices, default_values=self.default_values,
                                       first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

        return (len(existing_indices), len(missing_indices))
