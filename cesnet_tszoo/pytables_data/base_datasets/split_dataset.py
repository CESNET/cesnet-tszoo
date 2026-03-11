from typing import Any
from copy import deepcopy

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME, BASE_DATA_DTYPE_PART, TIME_DTYPE_PART
from cesnet_tszoo.pytables_data.base_datasets.base_dataset import BaseDataset
from cesnet_tszoo.utils.enums import TimeFormat


class SplitDataset(BaseDataset):
    """
    Split dataset base for main time based data loading... train, val, test etc.  

    Returns `batch_size` times for each time series in `ts_row_ranges`.
    """

    def __getitem__(self, batch_idx) -> Any:
        if batch_idx[0] == 0:
            self.load_config = deepcopy(self.saved_load_config)

        data = self.load_data_from_table(self.load_config.ts_row_ranges, self.load_config.time_period[batch_idx])

        if self.load_config.include_time:
            if self.load_config.time_format == TimeFormat.ID_TIME:
                data[BASE_DATA_DTYPE_PART][:, :, self.time_col_index] = self.load_config.time_period[batch_idx][ID_TIME_COLUMN_NAME]
            elif self.load_config.time_format == TimeFormat.UNIX_TIME or self.load_config.time_format == TimeFormat.SHIFTED_UNIX_TIME:
                data[BASE_DATA_DTYPE_PART][:, :, self.time_col_index] = self.load_config.time_period[batch_idx][TIME_COLUMN_NAME]
            elif self.load_config.time_format == TimeFormat.DATETIME:
                data[TIME_DTYPE_PART] = self.load_config.time_period[batch_idx][TIME_COLUMN_NAME]

        if self.load_config.include_ts_id:
            data[BASE_DATA_DTYPE_PART][:, :, self.ts_id_col_index] = self.ts_id_fill

        return data if len(self.load_config.return_dtype.names) > 1 else data[self.load_config.return_dtype.names[0]]

    def __len__(self) -> int:
        return len(self.load_config.time_period)
