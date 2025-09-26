import logging

from cesnet_tszoo.pytables_data.datasets.disjoint_time_based.split_dataset import DisjointTimeBasedSplitDataset
from cesnet_tszoo.pytables_data.base_datasets.splitted_dataset import SplittedDataset
from cesnet_tszoo.data_models.load_dataset_configs.time_load_config import TimeLoadConfig


class DisjointTimeBasedSplittedDataset(SplittedDataset):
    """
    Works as a wrapper around multiple/single DisjointTimeBasedSplitDataset. 

    Splits ts_row_ranges based on workers and for each worker creates a DisjointTimeBasedSplitDataset with subset of values from ts_row_ranges. Then each worker gets a dataloader.
    """

    def __init__(self, database_path: str, table_data_path: str, load_config: TimeLoadConfig, workers: int):
        self.load_config = load_config

        super().__init__(database_path, table_data_path, load_config, workers)

        self.logger = logging.getLogger("disjoint_time_based_splitted_dataset")

    def _create_dataset_split(self, split_range: slice) -> DisjointTimeBasedSplitDataset:
        split_load_config = self.load_config.create_split_copy(split_range)

        return DisjointTimeBasedSplitDataset(self.database_path, self.table_data_path, split_load_config)
