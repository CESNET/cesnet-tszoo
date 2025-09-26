from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset

from cesnet_tszoo.pytables_data.dataloaders import SeriesBasedDataloader, TimeBasedDataloader, DisjointTimeBasedDataloader
from cesnet_tszoo.pytables_data.time_based_splitted_dataset import TimeBasedSplittedDataset
from cesnet_tszoo.pytables_data.disjoint_time_based_splitted_dataset import DisjointTimeBasedSplittedDataset
from cesnet_tszoo.pytables_data.series_based_dataset import SeriesBasedDataset
from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.configs import SeriesBasedConfig, TimeBasedConfig, DisjointTimeBasedConfig
from cesnet_tszoo.utils.enums import DataloaderOrder


class DataloaderFactory(ABC):
    """Base class for dataloader factories. """

    @abstractmethod
    def create_dataloader(self, dataset: Dataset, dataset_config: DatasetConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> DataLoader:
        """Creates dataloader instance. """
        ...


# Implemented factories

class TimeBasedDataloaderFactory(DataloaderFactory):
    """Factory class for time based dataloader. """

    def create_dataloader(self, dataset: TimeBasedSplittedDataset, dataset_config: TimeBasedConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> TimeBasedDataloader:
        return TimeBasedDataloader(dataset, dataset_config, workers, take_all, batch_size)


class DisjointTimeBasedDataloaderFactory(DataloaderFactory):
    """Factory class for disjoint time based dataloader. """

    def create_dataloader(self, dataset: DisjointTimeBasedSplittedDataset, dataset_config: DisjointTimeBasedConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> DisjointTimeBasedDataloader:
        return DisjointTimeBasedDataloader(dataset, dataset_config, workers, take_all, batch_size)


class SeriesBasedDataloaderFactory(DataloaderFactory):
    """Factory class for series based dataloader. """

    def create_dataloader(self, dataset: SeriesBasedDataset, dataset_config: SeriesBasedConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> SeriesBasedDataloader:
        return SeriesBasedDataloader(dataset, dataset_config, workers, take_all, batch_size, order)

# Implemented factories
