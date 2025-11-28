from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset

from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.utils.enums import DataloaderOrder


class DataloaderFactory(ABC):
    """Base class for dataloader factories. """

    @abstractmethod
    def create_dataloader(self, dataset: Dataset, dataset_config: DatasetConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> DataLoader:
        """Creates dataloader instance. """
        ...
