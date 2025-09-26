from cesnet_tszoo.pytables_data.datasets.disjoint_time_based.splitted_dataset import DisjointTimeBasedSplittedDataset
from cesnet_tszoo.pytables_data.datasets.disjoint_time_based.dataloader import DisjointTimeBasedDataloader
from cesnet_tszoo.configs import DisjointTimeBasedConfig
from cesnet_tszoo.utils.enums import DataloaderOrder
from cesnet_tszoo.pytables_data.dataloader_factory import DataloaderFactory


class DisjointTimeBasedDataloaderFactory(DataloaderFactory):
    """Factory class for disjoint time based dataloader. """

    def create_dataloader(self, dataset: DisjointTimeBasedSplittedDataset, dataset_config: DisjointTimeBasedConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> DisjointTimeBasedDataloader:
        return DisjointTimeBasedDataloader(dataset, dataset_config, workers, take_all, batch_size)
