from copy import deepcopy
import logging

from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from cesnet_tszoo.pytables_data.disjoint_based_splitted_dataset import DisjointTimeBasedSplittedDataset
from cesnet_tszoo.configs.disjoint_time_based_config import DisjointTimeBasedConfig
import cesnet_tszoo.datasets.utils.loaders as dataset_loaders
from cesnet_tszoo.utils.constants import LOADING_WARNING_THRESHOLD


class DisjointTimeBasedDataloader(DataLoader):
    """Dataloader used for DisjointTimeBasedDataset. """

    def __init__(self, dataset: DisjointTimeBasedSplittedDataset, dataset_config: DisjointTimeBasedConfig, workers: int, take_all: bool, batch_size: int):

        dataset = deepcopy(dataset)
        self.logger = logging.getLogger('disjoint_time_based_dataloader')

        if take_all:
            batch_size = len(dataset)
            self.logger.debug("Using full dataset as batch size (%d samples).", batch_size)
        else:
            self.logger.debug("Using batch size from config: %d", batch_size)

            total_batch_size = batch_size * len(dataset.load_config.ts_row_ranges)
            if total_batch_size >= LOADING_WARNING_THRESHOLD:
                self.logger.warning("The total number of samples in one batch is %d (%d time series × %d times(batch size) ). Consider lowering the batch size.", total_batch_size, len(dataset.load_config.ts_row_ranges), batch_size)

        should_drop = not take_all and dataset_config.sliding_window_size is not None
        batch_sampler = BatchSampler(sampler=SequentialSampler(dataset), batch_size=batch_size, drop_last=should_drop)

        super().__init__(dataset, num_workers=0, collate_fn=dataset_loaders.collate_fn_simple, persistent_workers=False, batch_size=None, sampler=batch_sampler)

        self.logger.debug("Dataloader created with SequentialSampler and batch size %d.", batch_size)

        # Prepare the dataset for loading, either with the full batch or with windowed batching
        if take_all:
            dataset.prepare_dataset(batch_size, None, None, None, workers)
            self.logger.debug("DisjointTimeBasedSplittedDataset prepared with full batch size (%d samples).", batch_size)
        else:
            dataset.prepare_dataset(batch_size, dataset_config.sliding_window_size, dataset_config.sliding_window_prediction_size, dataset_config.sliding_window_step, workers)
            self.logger.debug("DisjointTimeBasedSplittedDataset prepared with window size (%d).", dataset_config.sliding_window_size)
