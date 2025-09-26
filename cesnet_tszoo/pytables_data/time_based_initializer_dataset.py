from copy import deepcopy

import numpy as np

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME
from cesnet_tszoo.pytables_data.base_datasets.initializer_dataset import InitializerDataset
from cesnet_tszoo.data_models.init_dataset_configs.time_init_config import TimeDatasetInitConfig


class TimeBasedInitializerDataset(InitializerDataset):
    """Used for time based datasets. Used for going through data to fit transformers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, init_config: TimeDatasetInitConfig):
        self.init_config = init_config

        super(TimeBasedInitializerDataset, self).__init__(database_path, table_data_path, init_config)

    def __getitem__(self, idx):

        data, count_values = self.load_data_from_table(self.init_config.ts_row_ranges[idx], idx)

        train_count_values, val_count_values, test_count_values, all_count_values = count_values

        this_val_filler = self.init_config.val_fillers[idx] if self.init_config.val_time_period is not None else None
        this_test_filler = self.init_config.test_fillers[idx] if self.init_config.test_time_period is not None else None
        this_anomaly_handler = self.init_config.anomaly_handlers[idx] if self.init_config.anomaly_handlers is not None else None

        # Prepare train data from current time series, if train set is used
        train_data = None
        if self.init_config.train_time_period is not None:
            if len(self.init_config.indices_of_features_to_take_no_ids) == 1:
                train_data = data[: len(self.init_config.train_time_period), self.offset_exclude_feature_ids:].reshape(-1, 1)
            elif len(self.init_config.train_time_period) == 1:
                train_data = data[: len(self.init_config.train_time_period), self.offset_exclude_feature_ids:].reshape(1, -1)
            else:
                train_data = data[: len(self.init_config.train_time_period), self.offset_exclude_feature_ids:]

        return train_data, train_count_values, val_count_values, test_count_values, all_count_values, this_val_filler, this_test_filler, this_anomaly_handler

    def __len__(self) -> int:
        return len(self.init_config.ts_row_ranges)

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
        if self.init_config.train_time_period is not None:
            first_start_index = offset_start = self.init_config.train_time_period[ID_TIME_COLUMN_NAME][0]
            train_existing_indices = np.where(missing_values_mask[:len(self.init_config.train_time_period)] == 0)[0]
            train_missing_indices = np.where(missing_values_mask[:len(self.init_config.train_time_period)] == 1)[0]

        # Missing/existing indices for validation set; Additionally prepares validation fillers based on previous values if needed and fills data
        if self.init_config.val_time_period is not None:

            current_start_index = self.init_config.val_time_period[ID_TIME_COLUMN_NAME][0]
            if first_start_index is not None and current_start_index > first_start_index:
                previous_offset = current_start_index - first_start_index

                previous_existing_indices = np.where(missing_values_mask[:current_start_index - first_start_index] == 0)[0]
                previous_missing_indices = np.where(missing_values_mask[:current_start_index - first_start_index] == 1)[0]

                train_should_fill = False
                self.init_config.val_fillers[idx].fill(result[:current_start_index - first_start_index, self.offset_exclude_feature_ids:].view(), previous_existing_indices, previous_missing_indices,
                                                       default_values=self.init_config.default_values,
                                                       first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

            offset_start = min(offset_start, self.init_config.val_time_period[ID_TIME_COLUMN_NAME][0])
            offsetted_val_time_period = self.init_config.val_time_period[ID_TIME_COLUMN_NAME] - offset_start

            val_existing_indices = np.where(missing_values_mask[offsetted_val_time_period] == 0)[0]
            val_missing_indices = np.where(missing_values_mask[offsetted_val_time_period] == 1)[0]

            if first_start_index is None:
                first_start_index = current_start_index

        # Missing/existing indices for test set; Additionally prepares test fillers based on previous values if needed and fills data
        if self.init_config.test_time_period is not None:

            current_start_index = self.init_config.test_time_period[ID_TIME_COLUMN_NAME][0]

            if first_start_index is not None and current_start_index > first_start_index:
                previous_existing_indices = np.where(missing_values_mask[previous_offset:current_start_index - first_start_index] == 0)[0]
                previous_missing_indices = np.where(missing_values_mask[previous_offset:current_start_index - first_start_index] == 1)[0]

                if self.init_config.val_time_period is not None:
                    self.init_config.test_fillers[idx] = deepcopy(self.init_config.val_fillers[idx])

                train_should_fill = False
                self.init_config.test_fillers[idx].fill(result[previous_offset:current_start_index - first_start_index, self.offset_exclude_feature_ids:].view(), previous_existing_indices, previous_missing_indices,
                                                        default_values=self.init_config.default_values,
                                                        first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

            offset_start = min(offset_start, self.init_config.test_time_period[ID_TIME_COLUMN_NAME][0])
            offsetted_test_time_period = self.init_config.test_time_period[ID_TIME_COLUMN_NAME] - offset_start

            test_existing_indices = np.where(missing_values_mask[offsetted_test_time_period] == 0)[0]
            test_missing_indices = np.where(missing_values_mask[offsetted_test_time_period] == 1)[0]

        if self.init_config.train_time_period is not None and train_should_fill:  # for transformer...
            self.init_config.train_fillers[idx].fill(result[:, self.offset_exclude_feature_ids:].view(), train_existing_indices, train_missing_indices,
                                                     default_values=self.init_config.default_values,
                                                     first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

        return (len(train_existing_indices), len(train_missing_indices)), (len(val_existing_indices), len(val_missing_indices)), (len(test_existing_indices), (len(test_missing_indices))), (len(all_existing_indices), (len(all_missing_indices)))

    def handle_anomalies(self, data: np.ndarray, idx: int):
        """Fits and uses anomaly handlers. """

        if self.init_config.anomaly_handlers is None:
            return

        self.init_config.anomaly_handlers[idx].fit(data[:len(self.init_config.train_time_period), self.offset_exclude_feature_ids:])
        self.init_config.anomaly_handlers[idx].transform_anomalies(data[:len(self.init_config.train_time_period), self.offset_exclude_feature_ids:])
