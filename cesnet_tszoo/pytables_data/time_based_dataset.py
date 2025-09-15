from typing import Any

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
from cesnet_tszoo.pytables_data.base_datasets.base_dataset import BaseDataset
from cesnet_tszoo.utils.enums import TimeFormat


class TimeBasedDataset(BaseDataset):
    """
    Used for main time based data loading... train, val, test etc.  

    Returns `batch_size` times for each time series in `ts_row_ranges`.
    """

    def __init__(self, database_path, table_data_path, ts_id_name, ts_row_ranges, time_period, features_to_take,
                 indices_of_features_to_take_no_ids, default_values, fillers, is_transformer_per_time_series, include_time, include_ts_id, time_format, transformers, anomaly_handlers):

        super().__init__(database_path, table_data_path, ts_id_name, ts_row_ranges, time_period, features_to_take,
                         indices_of_features_to_take_no_ids, default_values, fillers, include_time, include_ts_id, time_format, transformers, anomaly_handlers)

        self.is_transformer_per_time_series = is_transformer_per_time_series

    def __getitem__(self, batch_idx) -> Any:
        data = self.load_data_from_table(self.ts_row_ranges, self.time_period[batch_idx], self.fillers, self.anomaly_handlers)

        if self.include_time:
            if self.time_format == TimeFormat.ID_TIME:
                data[:, :, self.time_col_index] = self.time_period[batch_idx][ID_TIME_COLUMN_NAME]
            elif self.time_format == TimeFormat.UNIX_TIME or self.time_format == TimeFormat.SHIFTED_UNIX_TIME:
                data[:, :, self.time_col_index] = self.time_period[batch_idx][TIME_COLUMN_NAME]

        if self.include_ts_id:
            data[:, :, self.ts_id_col_index] = self.ts_id_fill

        # Transform data
        for i, _ in enumerate(self.ts_row_ranges):

            transformer = self.transformers[i] if self.is_transformer_per_time_series else self.transformers

            if len(self.indices_of_features_to_take_no_ids) == 1:
                data[i][:, self.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.indices_of_features_to_take_no_ids].reshape(-1, 1))
            elif len(batch_idx) == 1:
                data[i][:, self.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.indices_of_features_to_take_no_ids].reshape(1, -1))
            else:
                data[i][:, self.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.indices_of_features_to_take_no_ids])

        if self.include_time and self.time_format == TimeFormat.DATETIME:
            return data, self.time_period[batch_idx][TIME_COLUMN_NAME].copy()
        else:
            return data

    def __len__(self) -> int:
        return len(self.time_period)
