from threading import Thread
import logging

from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, Dataset
import numpy as np

from cesnet_tszoo.pytables_data.time_based_dataset import TimeBasedDataset
from cesnet_tszoo.utils.filler import Filler
from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.utils.anomaly_handler import AnomalyHandler
from cesnet_tszoo.utils.enums import TimeFormat


class SplittedDataset(Dataset):
    """
    Works as a wrapper around multiple/single TimeBasedDataset. 

    Splits ts_row_ranges based on workers and for each worker creates a TimeBasedDataset with subset of values from ts_row_ranges. Then each worker gets a dataloader.
    """

    def __init__(self, database_path: str, table_data_path: str, ts_id_name: str, ts_row_ranges: np.ndarray, time_period: np.ndarray, features_to_take: list[str], indices_of_features_to_take_no_ids: list[int],
                 default_values: np.ndarray, fillers: np.ndarray[Filler], is_transformer_per_time_series: bool,
                 include_time: bool, include_ts_id: bool, time_format: TimeFormat, workers: int, feature_transformers: np.ndarray[Transformer] | Transformer | None,
                 anomaly_handlers: np.ndarray[AnomalyHandler]):
        super().__init__()

        self.database_path = database_path
        self.table_data_path = table_data_path
        self.ts_id_name = ts_id_name
        self.features_to_take = features_to_take
        self.default_values = default_values
        self.fillers = fillers
        self.include_time = include_time
        self.include_ts_id = include_ts_id
        self.ts_row_ranges = ts_row_ranges

        self.logger = logging.getLogger("splitted_dataset")

        self.workers = min(workers, len(ts_row_ranges))
        if self.workers != workers:
            self.logger.debug("Using only %s workers because there is only %s datasets", self.workers, len(ts_row_ranges))

        self.workers_without_clip = workers
        self.time_period = time_period
        self.time_format = time_format
        self.feature_transformers = feature_transformers
        self.indices_of_features_to_take_no_ids = indices_of_features_to_take_no_ids
        self.is_transformer_per_time_series = is_transformer_per_time_series

        self.anomaly_handlers = anomaly_handlers

        self.datasets = []
        self.dataloaders = []
        self.dataloaders_iters = []
        self.iter_size = len(self.time_period)

        self.sliding_window_size = None
        self.sliding_window_prediction_size = None
        self.sliding_window_step = None
        self.data_for_window = None
        self.times_for_window = None
        self.until_next_batch_for_window = 0
        self.offset = 0
        self.batch_size = None

    def prepare_dataset(self, batch_size: int, sliding_window_size: int | None, sliding_window_prediction_size: int | None, sliding_window_step: int | None, workers: int) -> None:
        """Creates datasets and loaders based on workers. """

        self.logger.debug("Creating dataloaders for datasets.")

        self.batch_size = batch_size
        self.iter_size = len(self.time_period) if sliding_window_size is None else int((len(self.time_period) - sliding_window_size - (sliding_window_prediction_size - sliding_window_step)) * batch_size / sliding_window_step)
        self.sliding_window_size = sliding_window_size
        self.sliding_window_prediction_size = sliding_window_prediction_size
        self.sliding_window_step = sliding_window_step
        self.data_for_window = None
        self.times_for_window = None
        self.offset = 0
        self.until_next_batch_for_window = 0

        self.workers = min(workers, len(self.ts_row_ranges))
        if self.workers != workers:
            self.logger.debug("Using only %s workers because there is only %s datasets", self.workers, len(self.ts_row_ranges))
        self.workers_without_clip = workers

        self._reset()

        dataloader_workers = 0 if self.workers == 0 else 1
        dataloader_prefetch_factor = None if self.workers == 0 else 1

        for dataset in self.datasets:
            batch_sampler = BatchSampler(sampler=SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
            dataloader = DataLoader(
                dataset,
                num_workers=dataloader_workers,
                worker_init_fn=TimeBasedDataset.worker_init_fn,
                persistent_workers=False,
                batch_size=None,
                prefetch_factor=dataloader_prefetch_factor,
                sampler=batch_sampler, )

            self.dataloaders.append(dataloader)

            if self.workers == 0:
                dataset.pytables_worker_init(0)

    def _create_datasets(self):
        """Creates dataset for each worker. """

        self.logger.debug("Creating new datasets.")

        if self.workers > 0:
            sizes = np.zeros(self.workers, dtype=np.int32)
            sizes[:] += len(self.ts_row_ranges) // self.workers
            splits_with_additional = len(self.ts_row_ranges) % self.workers
            sizes[:splits_with_additional] += 1
        else:
            sizes = np.zeros(1, dtype=np.int32)
            sizes[0] = len(self.ts_row_ranges)

        offset = 0
        for size in sizes:

            transformers = None
            if self.feature_transformers is not None:
                transformers = self.feature_transformers[offset:offset + size] if self.is_transformer_per_time_series else self.feature_transformers

            fillers = None
            fillers = self.fillers[offset:offset + size]

            anomaly_handlers = None
            if self.anomaly_handlers is not None:
                anomaly_handlers = self.anomaly_handlers[offset:offset + size]

            dataset = TimeBasedDataset(self.database_path,
                                       self.table_data_path,
                                       self.ts_id_name,
                                       self.ts_row_ranges[offset:offset + size],
                                       self.time_period,
                                       self.features_to_take,
                                       self.indices_of_features_to_take_no_ids,
                                       self.default_values,
                                       fillers,
                                       self.is_transformer_per_time_series,
                                       self.include_time,
                                       self.include_ts_id,
                                       self.time_format,
                                       transformers,
                                       anomaly_handlers)
            self.datasets.append(dataset)
            offset += size

        self.logger.debug("Created %s datasets.", len(sizes))

    def __len__(self):
        return self.iter_size

    def _reset(self):
        """Cleans data. """

        self.dataloaders = []
        self.dataloaders_iters = []

        if self.workers != len(self.datasets) or len(self.datasets) == 0:
            self.datasets = []
            self._create_datasets()

    def __getitem__(self, batch_idx):

        if batch_idx[0] == 0:  # dataloader starts new iteration
            self.prepare_dataset(self.batch_size, self.sliding_window_size, self.sliding_window_prediction_size, self.sliding_window_step, self.workers)

        # Normal batch when window is not used
        if self.sliding_window_size is None:
            return self._get_data(batch_idx)

        # First window to return
        if self.data_for_window is None:
            if self.time_format == TimeFormat.DATETIME and self.include_time:
                self.data_for_window, self.times_for_window = self._get_data(batch_idx)
            else:
                self.data_for_window = self._get_data(batch_idx)

            self.offset = 0
            self.until_next_batch_for_window = self.data_for_window.shape[1] - self.sliding_window_size

        # Need more data for creating window
        elif self.until_next_batch_for_window < self.sliding_window_prediction_size:

            new_data_batch = None
            new_time_batch = None

            if self.time_format == TimeFormat.DATETIME and self.include_time:
                new_data_batch, new_time_batch = self._get_data(batch_idx)
            else:
                new_data_batch = self._get_data(batch_idx)

            self.data_for_window = np.concatenate([self.data_for_window[:, self.offset:, :], new_data_batch], axis=1)
            if self.time_format == TimeFormat.DATETIME and self.include_time:
                self.times_for_window = np.concatenate([self.times_for_window[self.offset:], new_time_batch], axis=0)

            self.offset = 0
            self.until_next_batch_for_window = self.data_for_window.shape[1] - self.sliding_window_size

        # Prepare data in window form
        if self.time_format == TimeFormat.DATETIME and self.include_time:
            result_data = (self.data_for_window[:, self.offset:self.offset + self.sliding_window_size, :], self.data_for_window[:, self.offset + self.sliding_window_size:self.offset + self.sliding_window_size + self.sliding_window_prediction_size, :].reshape((self.data_for_window.shape[0], self.sliding_window_prediction_size, self.data_for_window.shape[2])))
            result_time = (self.times_for_window[self.offset:self.offset + self.sliding_window_size], self.times_for_window[self.offset + self.sliding_window_size:self.offset + self.sliding_window_size + self.sliding_window_prediction_size])
        else:
            result_data = (self.data_for_window[:, self.offset:self.offset + self.sliding_window_size, :], self.data_for_window[:, self.offset + self.sliding_window_size:self.offset + self.sliding_window_size + self.sliding_window_prediction_size, :].reshape((self.data_for_window.shape[0], self.sliding_window_prediction_size, self.data_for_window.shape[2])))

        self.offset += self.sliding_window_step
        self.until_next_batch_for_window -= self.sliding_window_step

        if self.time_format == TimeFormat.DATETIME and self.include_time:
            return *result_data, *result_time
        else:
            return result_data

    def _get_data(self, batch_idx):
        """Returns concantated data from each dataset/worker."""

        batch_parts = []
        times = None

        if batch_idx[0] == 0:
            self.dataloaders_iters = []

            for dataloader in self.dataloaders:
                self.dataloaders_iters.append(iter(dataloader))

        workers = []
        results = [None for _ in self.datasets]

        def _process_dataset(worker_id, dataloader_iter):
            results[worker_id] = next(dataloader_iter)

        for i, dataloader_iter in enumerate(self.dataloaders_iters):
            worker = Thread(target=_process_dataset, args=(i, dataloader_iter))
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()

        for batch_part in results:
            if self.time_format == TimeFormat.DATETIME and self.include_time:
                batch_parts.append(batch_part[0])
                times = batch_part[1]
            else:
                batch_parts.append(batch_part)

        if self.time_format == TimeFormat.DATETIME and self.include_time:
            return np.concatenate(batch_parts, axis=0), times
        else:
            return np.concatenate(batch_parts, axis=0)
