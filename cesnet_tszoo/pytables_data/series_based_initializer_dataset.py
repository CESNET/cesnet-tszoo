import numpy as np

from cesnet_tszoo.pytables_data.base_datasets.initializer_dataset import InitializerDataset
from cesnet_tszoo.data_models.init_dataset_configs.series_init_config import SeriesDatasetInitConfig


class SeriesBasedInitializerDataset(InitializerDataset):
    """Used for series based datasets. Used for going through data to fit transformers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, init_config: SeriesDatasetInitConfig):
        self.init_config = init_config

        super(SeriesBasedInitializerDataset, self).__init__(database_path, table_data_path, init_config)

    def __getitem__(self, idx):

        data, count_values = self.load_data_from_table(self.init_config.ts_row_ranges[idx], idx)
        this_anomaly_handler = self.init_config.anomaly_handlers[idx] if self.init_config.anomaly_handlers is not None else None

        # Prepare data from current time series for training transformer if needed
        if len(self.init_config.indices_of_features_to_take_no_ids) == 1:
            train_data = data[: len(self.init_config.time_period), self.offset_exclude_feature_ids:].reshape(-1, 1)
        elif len(self.init_config.time_period) == 1:
            train_data = data[: len(self.init_config.time_period), self.offset_exclude_feature_ids:].reshape(1, -1)
        else:
            train_data = data[: len(self.init_config.time_period), self.offset_exclude_feature_ids:]

        return train_data, count_values, this_anomaly_handler

    def __len__(self) -> int:
        return len(self.init_config.ts_row_ranges)

    def fill_values(self, missing_values_mask: np.ndarray, idx, result, first_next_existing_values, first_next_existing_values_distance):
        """Just fills data. """

        existing_indices = np.where(missing_values_mask == 0)[0]
        missing_indices = np.where(missing_values_mask == 1)[0]

        self.init_config.fillers[idx].fill(result[:, self.offset_exclude_feature_ids:].view(), existing_indices, missing_indices, default_values=self.init_config.default_values,
                                           first_next_existing_values=first_next_existing_values, first_next_existing_values_distance=first_next_existing_values_distance)

        return (len(existing_indices), len(missing_indices))

    def handle_anomalies(self, data: np.ndarray, idx: int):
        """Fits and uses anomaly handlers. """

        if self.init_config.anomaly_handlers is None:
            return

        self.init_config.anomaly_handlers[idx].fit(data[:, self.offset_exclude_feature_ids:])
        self.init_config.anomaly_handlers[idx].transform_anomalies(data[:, self.offset_exclude_feature_ids:])
