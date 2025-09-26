from typing import Any

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
from cesnet_tszoo.pytables_data.base_datasets.base_dataset import BaseDataset
from cesnet_tszoo.data_models.load_dataset_configs.load_config import LoadConfig
from cesnet_tszoo.utils.enums import TimeFormat


class SplitDataset(BaseDataset):
    """
    Split dataset base for main time based data loading... train, val, test etc.  

    Returns `batch_size` times for each time series in `ts_row_ranges`.
    """

    def __init__(self, database_path: str, table_data_path: str, load_config: LoadConfig):
        super().__init__(database_path, table_data_path, load_config)

        self.is_transformer_per_time_series = load_config.is_transformer_per_time_series

    def __getitem__(self, batch_idx) -> Any:
        data = self.load_data_from_table(self.load_config.ts_row_ranges, self.load_config.time_period[batch_idx], self.load_config.fillers, self.load_config.anomaly_handlers)

        if self.load_config.include_time:
            if self.load_config.time_format == TimeFormat.ID_TIME:
                data[:, :, self.time_col_index] = self.load_config.time_period[batch_idx][ID_TIME_COLUMN_NAME]
            elif self.load_config.time_format == TimeFormat.UNIX_TIME or self.load_config.time_format == TimeFormat.SHIFTED_UNIX_TIME:
                data[:, :, self.time_col_index] = self.load_config.time_period[batch_idx][TIME_COLUMN_NAME]

        if self.load_config.include_ts_id:
            data[:, :, self.ts_id_col_index] = self.ts_id_fill

        # Transform data
        for i, _ in enumerate(self.load_config.ts_row_ranges):

            transformer = self.load_config.transformers[i] if self.is_transformer_per_time_series else self.load_config.transformers

            if len(self.load_config.indices_of_features_to_take_no_ids) == 1:
                data[i][:, self.load_config.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.load_config.indices_of_features_to_take_no_ids].reshape(-1, 1))
            elif len(batch_idx) == 1:
                data[i][:, self.load_config.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.load_config.indices_of_features_to_take_no_ids].reshape(1, -1))
            else:
                data[i][:, self.load_config.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.load_config.indices_of_features_to_take_no_ids])

        if self.load_config.include_time and self.load_config.time_format == TimeFormat.DATETIME:
            return data, self.load_config.time_period[batch_idx][TIME_COLUMN_NAME].copy()
        else:
            return data

    def __len__(self) -> int:
        return len(self.load_config.time_period)
