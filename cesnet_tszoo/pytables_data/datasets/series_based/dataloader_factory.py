from cesnet_tszoo.pytables_data.series_based_dataset import SeriesBasedDataset
from cesnet_tszoo.pytables_data.datasets.series_based.dataloader import SeriesBasedDataloader
from cesnet_tszoo.configs import SeriesBasedConfig
from cesnet_tszoo.utils.enums import DataloaderOrder
from cesnet_tszoo.pytables_data.dataloader_factory import DataloaderFactory


class SeriesBasedDataloaderFactory(DataloaderFactory):
    """Factory class for series based dataloader. """

    def create_dataloader(self, dataset: SeriesBasedDataset, dataset_config: SeriesBasedConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> SeriesBasedDataloader:
        return SeriesBasedDataloader(dataset, dataset_config, workers, take_all, batch_size, order)
