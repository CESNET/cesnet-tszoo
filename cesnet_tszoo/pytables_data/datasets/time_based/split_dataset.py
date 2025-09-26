from cesnet_tszoo.pytables_data.base_datasets.split_dataset import SplitDataset
from cesnet_tszoo.data_models.load_dataset_configs.time_load_config import TimeLoadConfig


class TimeSplitBasedDataset(SplitDataset):
    """
    Used for main time based data loading... train, val, test etc.  

    Returns `batch_size` times for each time series in `ts_row_ranges`.
    """

    def __init__(self, database_path: str, table_data_path: str, load_config: TimeLoadConfig):
        self.load_config = load_config

        super().__init__(database_path, table_data_path, load_config)
