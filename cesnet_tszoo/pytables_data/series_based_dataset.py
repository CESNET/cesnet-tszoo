from copy import deepcopy

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.pytables_data.base_datasets.base_dataset import BaseDataset


class SeriesBasedDataset(BaseDataset):
    """
    Used for main series based data loading... train, val, test etc.  

    Supports random batch indices and returns `batch_size` time series with times in `time_period`.
    """

    def __getitem__(self, batch_idx):
        if batch_idx[0] == 0:
            self.load_config = deepcopy(self.saved_load_config)

        data = self.load_data_from_table(self.load_config.ts_row_ranges[batch_idx], self.load_config.time_period)

        if self.load_config.include_time:
            if self.load_config.time_format == TimeFormat.ID_TIME:
                data[:, :, self.time_col_index] = self.load_config.time_period[ID_TIME_COLUMN_NAME]
            elif self.load_config.time_format == TimeFormat.UNIX_TIME or self.load_config.time_format == TimeFormat.SHIFTED_UNIX_TIME:
                data[:, :, self.time_col_index] = self.load_config.time_period[TIME_COLUMN_NAME]

        if self.load_config.include_ts_id:
            data[:, :, self.ts_id_col_index] = self.ts_id_fill[batch_idx]

        if self.load_config.include_time and self.load_config.time_format == TimeFormat.DATETIME:
            return data, self.load_config.time_period[TIME_COLUMN_NAME].copy()
        else:
            return data

    def __len__(self) -> int:
        return len(self.load_config.ts_row_ranges)
