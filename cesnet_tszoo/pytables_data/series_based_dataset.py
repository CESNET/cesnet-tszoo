from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.pytables_data.base_datasets.base_dataset import BaseDataset


class SeriesBasedDataset(BaseDataset):
    """
    Used for main series based data loading... train, val, test etc.  

    Supports random batch indices and returns `batch_size` time series with times in `time_period`.
    """

    def __getitem__(self, batch_idx):
        fillers = self.fillers[batch_idx] if self.fillers is not None else None

        data = self.load_data_from_table(self.ts_row_ranges[batch_idx], self.time_period, fillers)

        if self.include_time:
            if self.time_format == TimeFormat.ID_TIME:
                data[:, :, self.time_col_index] = self.time_period[ID_TIME_COLUMN_NAME]
            elif self.time_format == TimeFormat.UNIX_TIME or self.time_format == TimeFormat.SHIFTED_UNIX_TIME:
                data[:, :, self.time_col_index] = self.time_period[TIME_COLUMN_NAME]

        if self.include_ts_id:
            data[:, :, self.ts_id_col_index] = self.ts_id_fill[batch_idx]

        # Transform data if applicable
        if self.transformers is not None:
            transformer = self.transformers
            for i, _ in enumerate(data):
                if len(self.indices_of_features_to_take_no_ids) == 1:
                    data[i][:, self.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.indices_of_features_to_take_no_ids].reshape(-1, 1))
                elif len(self.time_period) == 1:
                    data[i][:, self.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.indices_of_features_to_take_no_ids].reshape(1, -1))
                else:
                    data[i][:, self.indices_of_features_to_take_no_ids] = transformer.transform(data[i][:, self.indices_of_features_to_take_no_ids])

        if self.include_time and self.time_format == TimeFormat.DATETIME:
            return data, self.time_period[TIME_COLUMN_NAME].copy()
        else:
            return data

    def __len__(self) -> int:
        return len(self.ts_row_ranges)
