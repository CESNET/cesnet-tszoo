from cesnet_tszoo.pytables_data.time_based_splitted_dataset import TimeBasedSplittedDataset
from cesnet_tszoo.pytables_data.datasets.time_based.dataloader import TimeBasedDataloader
from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.utils.enums import DataloaderOrder
from cesnet_tszoo.pytables_data.dataloader_factory import DataloaderFactory


class TimeBasedDataloaderFactory(DataloaderFactory):
    """Factory class for time based dataloader. """

    def create_dataloader(self, dataset: TimeBasedSplittedDataset, dataset_config: TimeBasedConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> TimeBasedDataloader:
        return TimeBasedDataloader(dataset, dataset_config, workers, take_all, batch_size)
