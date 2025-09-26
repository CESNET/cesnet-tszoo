from copy import deepcopy
import logging

from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
import torch

from cesnet_tszoo.pytables_data.series_based_dataset import SeriesBasedDataset
from cesnet_tszoo.configs.series_based_config import SeriesBasedConfig
from cesnet_tszoo.utils.enums import DataloaderOrder
import cesnet_tszoo.datasets.utils.loaders as dataset_loaders
from cesnet_tszoo.utils.constants import LOADING_WARNING_THRESHOLD


class SeriesBasedDataloader(DataLoader):
    """Dataloader used for TimeBasedDataset. """

    def __init__(self, dataset: SeriesBasedDataset, dataset_config: SeriesBasedConfig, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL):

        dataset = deepcopy(dataset)
        self.logger = logging.getLogger('series_based_dataloader')

        if take_all:
            batch_size = len(dataset)
            self.logger.debug("Using full dataset as batch size (%d samples) to return the entire dataset.", batch_size)
        else:
            self.logger.debug("Using batch size from config: %d", batch_size)

        total_batch_size = batch_size * len(dataset.time_period)
        if total_batch_size >= LOADING_WARNING_THRESHOLD:
            self.logger.warning("The total number of samples in one batch is %d (%d time series(batch size) × %d times ). Consider lowering the batch size.", total_batch_size, batch_size, len(dataset.time_period))

        if order == DataloaderOrder.RANDOM:
            if dataset_config.random_state is not None:
                generator = torch.Generator()
                generator.manual_seed(dataset_config.random_state)
                self.logger.debug("Prepared RandomSampler with fixed seed %d for series dataloader.", dataset_config.random_state)
            else:
                generator = None
                self.logger.debug("Prepared RandomSampler with dynamic seed for series dataloader.")

            sampler = RandomSampler(dataset, generator=generator)

        elif order == DataloaderOrder.SEQUENTIAL:
            sampler = SequentialSampler(dataset)
            self.logger.debug("Prepared SequentialSampler for series dataloader.")
        else:
            raise ValueError("Invalid order specified for the dataloader. Supported values are DataloaderOrder.RANDOM and DataloaderOrder.SEQUENTIAL.")

        batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)

        super().__init__(dataset, num_workers=workers, collate_fn=dataset_loaders.collate_fn_simple, worker_init_fn=SeriesBasedDataset.worker_init_fn, persistent_workers=False, batch_size=None, sampler=batch_sampler)

        # Must be done if dataloader runs on main process.
        if workers == 0:
            dataset.pytables_worker_init(0)

        self.logger.debug("SeriesBasedDataset prepared for dataloader with batch size %d and %s order.", batch_size, order.name)
