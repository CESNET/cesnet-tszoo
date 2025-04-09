from copy import deepcopy

import numpy as np

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME
from cesnet_tszoo.utils.filler import Filler
from cesnet_tszoo.pytables_data.base_datasets.initializer_dataset import InitializerDataset


class TimeBasedInitializerDataset(InitializerDataset):
    """Used for time based datasets. Used for going through data to fit scalers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, ts_id_name: str, ts_row_ranges: np.ndarray, all_time_period: np.ndarray, train_time_period: np.ndarray, val_time_period: np.ndarray, test_time_period: np.ndarray,
                 features_to_take: list[str], indices_of_features_to_take_no_ids: list[int], default_values: np.ndarray, train_fillers: np.ndarray[Filler], val_fillers: np.ndarray[Filler], test_fillers: np.ndarray[Filler]):
        self.ts_row_ranges = ts_row_ranges

        self.train_time_period = train_time_period
        self.val_time_period = val_time_period
        self.test_time_period = test_time_period
        self.all_time_period = all_time_period

        self.train_fillers = deepcopy(train_fillers)
        self.val_fillers = val_fillers
        self.test_fillers = test_fillers

        time_period = None

        if self.all_time_period is None:
            for temp_time_period in [self.train_time_period, self.val_time_period, self.test_time_period]:
                if temp_time_period is None:
                    continue
                elif time_period is None:
                    time_period = temp_time_period.copy()
                else:
                    time_period = np.concatenate((time_period, temp_time_period))

            time_period = np.unique(time_period)
        else:
            time_period = self.all_time_period.copy()

        super(TimeBasedInitializerDataset, self).__init__(database_path, table_data_path, ts_id_name, time_period, features_to_take, indices_of_features_to_take_no_ids, default_values)

    def __getitem__(self, idx):

        data, count_values = self.load_data_from_table(self.ts_row_ranges[idx], idx)

        train_count_values, val_count_values, test_count_values, all_count_values = count_values

        this_val_filler = self.val_fillers[idx] if self.val_time_period is not None and self.val_fillers is not None else None
        this_test_filler = self.test_fillers[idx] if self.test_time_period is not None and self.test_fillers is not None else None

        # Prepare train data from current time series, if train set is used
        train_data = None
        if self.train_time_period is not None:
            if len(self.indices_of_features_to_take_no_ids) == 1:
                train_data = data[: len(self.train_time_period), self.offset_exclude_feature_ids:].reshape(-1, 1)
            elif len(self.train_time_period) == 1:
                train_data = data[: len(self.train_time_period), self.offset_exclude_feature_ids:].reshape(1, -1)
            else:
                train_data = data[: len(self.train_time_period), self.offset_exclude_feature_ids:]

        return train_data, train_count_values, val_count_values, test_count_values, all_count_values, this_val_filler, this_test_filler

    def __len__(self) -> int:
        return len(self.ts_row_ranges)

    def fill_values(self, missing_values_mask: np.ndarray, idx, result, first_next_existing_values, first_next_existing_values_distance):
        """Fills data and prepares fillers based on previous times. Order is train (does not need to update train fillers) > val > test."""

        train_existing_indices = []
        train_missing_indices = []
        val_existing_indices = []
        val_missing_indices = []
        test_existing_indices = []
        test_missing_indices = []
        all_existing_indices = np.where(missing_values_mask == 0)[0]
        all_missing_indices = np.where(missing_values_mask == 1)[0]

        offset_start = np.inf
        first_start_index = None
        train_should_fill = True
        previous_offset = 0

        # Missing/existing indices for train set
        if self.train_time_period is not None:
            first_start_index = offset_start = self.train_time_period[ID_TIME_COLUMN_NAME][0]
            train_existing_indices = np.where(missing_values_mask[:len(self.train_time_period)] == 0)[0]
            train_missing_indices = np.where(missing_values_mask[:len(self.train_time_period)] == 1)[0]

        # Missing/existing indices for validation set; Additionally prepares validation fillers based on previous values if needed and fills data
        if self.val_time_period is not None:

            current_start_index = self.val_time_period[ID_TIME_COLUMN_NAME][0]
            if first_start_index is not None and current_start_index > first_start_index:
                previous_offset = current_start_index - first_start_index

                previous_existing_indices = np.where(missing_values_mask[:current_start_index - first_start_index] == 0)[0]
                previous_missing_indices = np.where(missing_values_mask[:current_start_index - first_start_index] == 1)[0]

                if self.val_fillers is not None:
                    train_should_fill = False
                    self.val_fillers[idx].fill(result[:current_start_index - first_start_index, self.offset_exclude_feature_ids:].view(), previous_existing_indices, previous_missing_indices,
                                               default_values=self.default_values,
                                               first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

            offset_start = min(offset_start, self.val_time_period[ID_TIME_COLUMN_NAME][0])
            offsetted_val_time_period = self.val_time_period[ID_TIME_COLUMN_NAME] - offset_start

            val_existing_indices = np.where(missing_values_mask[offsetted_val_time_period] == 0)[0]
            val_missing_indices = np.where(missing_values_mask[offsetted_val_time_period] == 1)[0]

            if first_start_index is None:
                first_start_index = current_start_index

        # Missing/existing indices for test set; Additionally prepares test fillers based on previous values if needed and fills data
        if self.test_time_period is not None:

            current_start_index = self.test_time_period[ID_TIME_COLUMN_NAME][0]

            if first_start_index is not None and current_start_index > first_start_index:
                previous_existing_indices = np.where(missing_values_mask[previous_offset:current_start_index - first_start_index] == 0)[0]
                previous_missing_indices = np.where(missing_values_mask[previous_offset:current_start_index - first_start_index] == 1)[0]

                if self.test_fillers is not None:
                    if self.val_fillers is not None:
                        self.test_fillers[idx] = deepcopy(self.val_fillers[idx])
                    train_should_fill = False
                    self.test_fillers[idx].fill(result[previous_offset:current_start_index - first_start_index, self.offset_exclude_feature_ids:].view(), previous_existing_indices, previous_missing_indices,
                                                default_values=self.default_values,
                                                first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

            offset_start = min(offset_start, self.test_time_period[ID_TIME_COLUMN_NAME][0])
            offsetted_test_time_period = self.test_time_period[ID_TIME_COLUMN_NAME] - offset_start

            test_existing_indices = np.where(missing_values_mask[offsetted_test_time_period] == 0)[0]
            test_missing_indices = np.where(missing_values_mask[offsetted_test_time_period] == 1)[0]

        if self.train_time_period is not None and self.train_fillers is not None and train_should_fill:  # for scaler...
            self.train_fillers[idx].fill(result[:, self.offset_exclude_feature_ids:].view(), train_existing_indices, train_missing_indices,
                                         default_values=self.default_values,
                                         first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

        return (len(train_existing_indices), len(train_missing_indices)), (len(val_existing_indices), len(val_missing_indices)), (len(test_existing_indices), (len(test_missing_indices))), (len(all_existing_indices), (len(all_missing_indices)))
