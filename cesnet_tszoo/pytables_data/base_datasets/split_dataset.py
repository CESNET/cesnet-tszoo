from typing import Any
from copy import deepcopy

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
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
                data[:, :, self.time_col_index] = self.load_config.time_period[batch_idx][ID_TIME_COLUMN_NAME]
            elif self.load_config.time_format == TimeFormat.UNIX_TIME or self.load_config.time_format == TimeFormat.SHIFTED_UNIX_TIME:
                data[:, :, self.time_col_index] = self.load_config.time_period[batch_idx][TIME_COLUMN_NAME]

        if self.load_config.include_ts_id:
            data[:, :, self.ts_id_col_index] = self.ts_id_fill

        if self.load_config.include_time and self.load_config.time_format == TimeFormat.DATETIME:
            return data, self.load_config.time_period[batch_idx][TIME_COLUMN_NAME].copy()
        else:
            return data

    def __len__(self) -> int:
        return len(self.load_config.time_period)
