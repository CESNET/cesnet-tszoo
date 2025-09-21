import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable, Literal
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass, field
from numbers import Number

import numpy as np
import numpy.typing as npt
import pandas as pd
import tables as tb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, Dataset, RandomSampler
import torch

import cesnet_tszoo.version as version
from cesnet_tszoo.files.utils import get_annotations_path_and_whether_it_is_built_in, exists_built_in_annotations, exists_built_in_benchmark, exists_built_in_config
import cesnet_tszoo.utils.filler.factory as filler_factories
import cesnet_tszoo.utils.transformer.factory as transformer_factories
import cesnet_tszoo.utils.anomaly_handler.factory as anomaly_handler_factories
from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.annotation import Annotations
import cesnet_tszoo.datasets.utils.loaders as dataset_loaders
from cesnet_tszoo.pytables_data.series_based_dataset import SeriesBasedDataset
from cesnet_tszoo.pytables_data.splitted_dataset import SplittedDataset
from cesnet_tszoo.pytables_data.utils.utils import get_additional_data, load_database
from cesnet_tszoo.utils.file_utils import pickle_dump, yaml_dump
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, LOADING_WARNING_THRESHOLD, ANNOTATIONS_DOWNLOAD_BUCKET
from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.utils.enums import SplitType, TimeFormat, DataloaderOrder, AnnotationType, FillerType, TransformerType, DatasetType, AnomalyHandlerType
from cesnet_tszoo.utils.utils import get_abbreviated_list_string, ExportBenchmark
from cesnet_tszoo.configs.handlers.time_based_handler import TimeBasedHandler
from cesnet_tszoo.utils.download import resumable_download
from cesnet_tszoo.configs.config_loading import load_config
from cesnet_tszoo.dataclasses.dataset_metadata import DatasetMetadata


@dataclass
class CesnetDataset(ABC):
    """
    Base class for cesnet datasets. This class should **not** be used directly. Instead, use one of the derived classes, such as [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset], [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] or [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset].

    The dataset provides multiple ways to access the data:

    - **Iterable PyTorch DataLoader**: For batch processing.
    - **Pandas DataFrame**: For loading the entire training, validation, test or all set at once.
    - **Numpy array**: For loading the entire training, validation, test or all set at once.    
    - See [loading data][loading-data] for more details.

    The dataset is stored in a [PyTables](https://www.pytables.org/) database. The internal `BaseDataset` and `InitializerDataset` classes (used only when calling [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize]) act as wrappers that implement the PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 
    interface. These wrappers are compatible with PyTorch’s [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), providing efficient parallel data loading. 

    The dataset configuration is done through the [`DatasetConfig`][cesnet_tszoo.configs.base_config.DatasetConfig] class.    

    **Intended usage:**

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`DatasetConfig`][cesnet_tszoo.configs.base_config.DatasetConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_test_numpy].     

    Alternatively you can use [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark]
    1. Call [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark] with the desired benchmark. You can use your own saved benchmark or you can use already built-in one. This will download the dataset and annotations (if available) if they have not been previously downloaded.
    2. Retrieve the initialized dataset using [`get_initialized_dataset`][cesnet_tszoo.benchmarks.Benchmark.get_initialized_dataset]. This will provide a dataset that is ready to use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_test_numpy].   

    Parameters:
        metadata: Holds various metadata used in dataset for its creation, loading data and similar.

    Attributes:
        metadata: Holds various metadata used in dataset for its creation, loading data and similar.
        annotations: Annotations for the selected dataset.
        logger: Logger for displaying information.  
        imported_annotations_ts_identifier: Identifier for the imported annotations of type `AnnotationType.TS_ID`.
        imported_annotations_time_identifier: Identifier for the imported annotations of type `AnnotationType.ID_TIME`.
        imported_annotations_both_identifier: Identifier for the imported annotations of type `AnnotationType.BOTH`.            

    The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize] is called:
    Attributes:
        dataset_config: Configuration of the dataset.
        train_dataset: Training set as a `BaseDataset` instance wrapping the PyTables database.
        val_dataset: Validation set as a `BaseDataset` instance wrapping the PyTables database.
        test_dataset: Test set as a `BaseDataset` instance wrapping the PyTables database.
        all_dataset: All set as a `BaseDataset` instance wrapping the PyTables database.
        train_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training set.
        val_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation set.
        test_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test set.
        all_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for all set.
    """

    metadata: DatasetMetadata

    dataset_config: Optional[DatasetConfig] = field(default=None, init=False)

    train_dataset: Optional[Dataset] = field(default=None, init=False)
    val_dataset: Optional[Dataset] = field(default=None, init=False)
    test_dataset: Optional[Dataset] = field(default=None, init=False)
    all_dataset: Optional[Dataset] = field(default=None, init=False)

    train_dataloader: Optional[DataLoader] = field(default=None, init=False)
    val_dataloader: Optional[DataLoader] = field(default=None, init=False)
    test_dataloader: Optional[DataLoader] = field(default=None, init=False)
    all_dataloader: Optional[DataLoader] = field(default=None, init=False)

    _collate_fn: Optional[Callable] = field(default=None, init=False)
    _export_config_copy: Optional[DatasetConfig] = field(default=None, init=False)

    def __post_init__(self):
        self.logger = logging.getLogger("cesnet_dataset")

        self._collate_fn = dataset_loaders.collate_fn_simple
        self.annotations = Annotations()

        # Initialize annotation states
        self.imported_annotations_ts_identifier = None
        self.imported_annotations_time_identifier = None
        self.imported_annotations_both_identifier = None

    def set_dataset_config_and_initialize(self, dataset_config: DatasetConfig, display_config_details: bool = True, workers: int | Literal["config"] = "config") -> None:
        """
        Initialize training set, validation set, test set etc.. This method must be called before any data can be accessed. It is required for the final initialization of [`dataset_config`][cesnet_tszoo.configs.base_config.DatasetConfig].

        The following configuration attributes are used during initialization:

        | Dataset config                         | Description                                                                                    |
        | -------------------------------------- | ---------------------------------------------------------------------------------------------- |
        | `init_workers`                         | Specifies the number of workers to use for initialization. Applied when `workers` = "config".  |
        | `partial_fit_initialized_transformers` | Determines whether initialized transformers should be partially fitted on the training data.   |
        | `nan_threshold`                        | Filters out time series with missing values exceeding the specified threshold.                 |

        Parameters:
            dataset_config: Desired configuration of the dataset.
            display_config_details: Flag indicating whether to display the configuration values after initialization. `Default: True`  
            workers: The number of workers to use during initialization. `Default: "config"`  
        """

        self._clear()
        self.dataset_config = dataset_config

        # If the config is not initialized, set a copy of the configuration for export
        if not self.dataset_config.is_initialized:
            self.dataset_config.aggregation = self.metadata.aggregation
            self.dataset_config.source_type = self.metadata.source_type
            self.dataset_config.database_name = self.metadata.database_name
            self._export_config_copy = deepcopy(self.dataset_config)
            self.logger.debug("New export_config_copy created.")

        self._validate_config_for_dataset(self.dataset_config)

        if workers == "config":
            workers = self.dataset_config.init_workers

        if not self.dataset_config.is_initialized:
            self.dataset_config._dataset_init(self.metadata)
            self._initialize_transformers_and_details(workers)
            self.dataset_config.is_initialized = True
            self.logger.info("Config initialized successfully.")
        else:
            self.logger.info("Config already initialized. Skipping re-initialization.")

        # Initialize datasets
        self._initialize_datasets()
        self.logger.debug("Datasets have been successfully initialized.")

        self._update_export_config_copy()
        self.logger.debug("Export config copy updated with the latest dataset configuration.")

        if display_config_details:
            self.display_config()

    def get_train_dataloader(self, ts_id: int | None = None, workers: int | Literal["config"] = "config", **kwargs) -> DataLoader:
        """
        Returns a PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training set.

        The `DataLoader` is created on the first call and cached for subsequent use. <br/>
        The cached dataloader is cleared when either [`get_train_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_train_df] or [`get_train_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_train_numpy] is called.

        The structure of the returned batch depends on the `time_format` and whether `sliding_window_size` is used:

        - When `sliding_window_size` is used:
            - With `time_format` == TimeFormat.DATETIME and included time:
                - `np.ndarray` of shape `(num_time_series, times - 1, features)`
                - `np.ndarray` of shape `(num_time_series, 1, features)`
                - `np.ndarray` of times with shape `(times - 1)`
                - `np.ndarray` of time with shape `(1)`
            - When `time_format` != TimeFormat.DATETIME or time is not included:
                - `np.ndarray` of shape `(num_time_series, times - 1, features)`
                - `np.ndarray` of shape `(num_time_series, 1, features)`
        - When `sliding_window_size` is not used:
            - With `time_format` == TimeFormat.DATETIME and included time:
                - `np.ndarray` of shape `(num_time_series, times, features)`
                - `np.ndarray` of time with shape `(times)`
            - When `time_format` != TimeFormat.DATETIME or time is not included:
                - `np.ndarray` of shape `(num_time_series, times, features)`

        The `DataLoader` is configured with the following config attributes:

        | Dataset config                    | Description                                                                                                                                            |
        | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
        | `train_batch_size`                | Number of samples per batch. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.                      |
        | `sliding_window_size`             | Available only for time-based datasets. Modifies the shape of the returned data.                                                                       |
        | `sliding_window_prediction_size`  | Available only for time-based datasets. Modifies the shape of the returned data.                                                                       |
        | `sliding_window_step`             | Available only for time-based datasets. Number of times to move by after each window.                                                     |
        | `train_workers`                   | Specifies the number of workers to use for loading train data. Applied when `workers` = "config".                                                      |
        | `train_dataloader_order`          | Available only for series-based datasets. Whether to load train data in sequential or random order. See [cesnet_tszoo.utils.enums.DataloaderOrder][].  |
        | `random_state`                    | Seed for loading train data in random order.                                                                                                           |                 

        Parameters:
            workers: The number of workers to use for loading train data. `Default: "config"` 
            ts_id: Specifies time series to take. If None returns all time series as normal. `Default: "None"`

        Returns:
            An iterable `DataLoader` containing data from training set.          
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access train_dataloader.")

        if not self.dataset_config.has_train():
            raise ValueError("Dataloader for training set is not available in the dataset configuration.")

        assert self.train_dataset is not None, "The train_dataset must be initialized before accessing data from training set."

        default_kwargs = {'take_all': False, "cache_loader": True}
        kwargs = {**default_kwargs, **kwargs}

        if ts_id is not None:

            if ts_id == self.dataset_config.used_singular_train_time_series and self.train_dataloader is not None:
                self.logger.debug("Returning cached train_dataloader.")
                return self.train_dataloader

            dataset = self._get_singular_time_series_dataset(self.train_dataset, ts_id)
            self.dataset_config.used_singular_train_time_series = ts_id
            if self.train_dataloader:
                del self.train_dataloader
                self.train_dataloader = None
                self.logger.info("Destroyed previous cached train_dataloader.")

            self.dataset_config.used_train_workers = 0
            self.train_dataloader = self._get_dataloader(dataset, 0, False, self.dataset_config.train_batch_size)
            self.logger.info("Created new cached train_dataloader.")
            return self.train_dataloader
        elif self.dataset_config.used_singular_train_time_series is not None and self.train_dataloader is not None:
            del self.train_dataloader
            self.train_dataloader = None
            self.dataset_config.used_singular_train_time_series = None
            self.logger.info("Destroyed previous cached train_dataloader.")

        if workers == "config":
            workers = self.dataset_config.train_workers

        # If the dataloader is cached and number of used workers did not change, return the cached dataloader
        if self.train_dataloader and kwargs["cache_loader"] and workers == self.dataset_config.used_train_workers:
            self.logger.debug("Returning cached train_dataloader.")
            return self.train_dataloader

        # Update the used workers count
        self.dataset_config.used_train_workers = workers

        # If there's a previously cached dataloader, destroy it
        if self.train_dataloader:
            del self.train_dataloader
            self.train_dataloader = None
            self.logger.info("Destroyed previous cached train_dataloader.")

        # If caching is enabled, create a new cached dataloader
        if kwargs["cache_loader"]:
            self.train_dataloader = self._get_dataloader(self.train_dataset, workers, kwargs['take_all'], self.dataset_config.train_batch_size, order=self.dataset_config.train_dataloader_order)
            self.logger.info("Created new cached train_dataloader.")
            return self.train_dataloader

        # If caching is disabled, create a new uncached dataloader
        self.logger.debug("Created new uncached train_dataloader.")
        return self._get_dataloader(self.train_dataset, workers, kwargs['take_all'], self.dataset_config.train_batch_size, order=self.dataset_config.train_dataloader_order)

    def get_val_dataloader(self, ts_id: int | None = None, workers: int | Literal["config"] = "config", **kwargs) -> DataLoader:
        """
        Returns a PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation set.

        The `DataLoader` is created on the first call and cached for subsequent use. <br/>
        The cached dataloader is cleared when either [`get_val_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_val_df] or [`get_val_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_val_numpy] is called.

        The structure of the returned batch depends on the `time_format` and whether `sliding_window_size` is used:

        - When `sliding_window_size` is used:
            - With `time_format` == TimeFormat.DATETIME and included time:
                - `np.ndarray` of shape `(num_time_series, times - 1, features)`
                - `np.ndarray` of shape `(num_time_series, 1, features)`
                - `np.ndarray` of times with shape `(times - 1)`
                - `np.ndarray` of time with shape `(1)`
            - When `time_format` != TimeFormat.DATETIME or time is not included:
                - `np.ndarray` of shape `(num_time_series, times - 1, features)`
                - `np.ndarray` of shape `(num_time_series, 1, features)`
        - When `sliding_window_size` is not used:
            - With `time_format` == TimeFormat.DATETIME and included time:
                - `np.ndarray` of shape `(num_time_series, times, features)`
                - `np.ndarray` of time with shape `(times)`
            - When `time_format` != TimeFormat.DATETIME or time is not included:
                - `np.ndarray` of shape `(num_time_series, times, features)`

        The `DataLoader` is configured with the following config attributes:

        | Dataset config                    | Description                                                                                                                               |
        | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
        | `val_batch_size`                  | Number of samples per batch. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.         |
        | `sliding_window_size`             | Available only for time-based datasets. Modifies the shape of the returned data.                                                          |
        | `sliding_window_prediction_size`  | Available only for time-based datasets. Modifies the shape of the returned data.                                                          |
        | `sliding_window_step`             | Available only for time-based datasets. Number of times to move by after each window.                                                     |
        | `val_workers`                     | Specifies the number of workers to use for loading validation data. Applied when `workers` = "config".                                    |

        Parameters:
            workers: The number of workers to use for loading validation data. `Default: "config"`  
            ts_id: Specifies time series to take. If None returns all time series as normal. `Default: "None"`

        Returns:
            An iterable `DataLoader` containing data from validation set.        
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access val_dataloader.")

        if not self.dataset_config.has_val():
            raise ValueError("Dataloader for validation set is not available in the dataset configuration.")

        assert self.val_dataset is not None, "The val_dataset must be initialized before accessing data from validation set."

        default_kwargs = {'take_all': False, "cache_loader": True}
        kwargs = {**default_kwargs, **kwargs}

        if ts_id is not None:

            if ts_id == self.dataset_config.used_singular_val_time_series and self.val_dataloader is not None:
                self.logger.debug("Returning cached val_dataloader.")
                return self.val_dataloader

            dataset = self._get_singular_time_series_dataset(self.val_dataset, ts_id)
            self.dataset_config.used_singular_val_time_series = ts_id
            if self.val_dataloader:
                del self.val_dataloader
                self.val_dataloader = None
                self.logger.info("Destroyed previous cached val_dataloader.")

            self.dataset_config.used_val_workers = 0
            self.val_dataloader = self._get_dataloader(dataset, 0, False, self.dataset_config.val_batch_size)
            self.logger.info("Created new cached val_dataloader.")
            return self.val_dataloader
        elif self.dataset_config.used_singular_val_time_series is not None and self.val_dataloader is not None:
            del self.val_dataloader
            self.val_dataloader = None
            self.dataset_config.used_singular_val_time_series = None
            self.logger.info("Destroyed previous cached val_dataloader.")

        if workers == "config":
            workers = self.dataset_config.val_workers

        # If the dataloader is cached and number of used workers did not change, return the cached dataloader
        if self.val_dataloader and kwargs["cache_loader"] and workers == self.dataset_config.used_val_workers:
            self.logger.debug("Returning cached val_dataloader.")
            return self.val_dataloader

        # Update the used workers count
        self.dataset_config.used_val_workers = workers

        # If there's a previously cached dataloader, destroy it
        if self.val_dataloader:
            del self.val_dataloader
            self.val_dataloader = None
            self.logger.info("Destroyed previous cached val_dataloader.")

        # If caching is enabled, create a new cached dataloader
        if kwargs["cache_loader"]:
            self.val_dataloader = self._get_dataloader(self.val_dataset, workers, kwargs['take_all'], self.dataset_config.val_batch_size)
            self.logger.info("Created new cached val_dataloader.")
            return self.val_dataloader

        # If caching is disabled, create a new uncached dataloader
        self.logger.debug("Created new uncached val_dataloader.")
        return self._get_dataloader(self.val_dataset, workers, kwargs['take_all'], self.dataset_config.val_batch_size)

    def get_test_dataloader(self, ts_id: int | None = None, workers: int | Literal["config"] = "config", **kwargs) -> DataLoader:
        """
        Returns a PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test set.

        The `DataLoader` is created on the first call and cached for subsequent use. <br/>
        The cached dataloader is cleared when either [`get_test_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_test_df] or [`get_test_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_test_numpy] is called.

        The structure of the returned batch depends on the `time_format` and whether `sliding_window_size` is used:

        - When `sliding_window_size` is used:
            - With `time_format` == TimeFormat.DATETIME and included time:
                - `np.ndarray` of shape `(num_time_series, times - 1, features)`
                - `np.ndarray` of shape `(num_time_series, 1, features)`
                - `np.ndarray` of times with shape `(times - 1)`
                - `np.ndarray` of time with shape `(1)`
            - When `time_format` != TimeFormat.DATETIME or time is not included:
                - `np.ndarray` of shape `(num_time_series, times - 1, features)`
                - `np.ndarray` of shape `(num_time_series, 1, features)`
        - When `sliding_window_size` is not used:
            - With `time_format` == TimeFormat.DATETIME and included time:
                - `np.ndarray` of shape `(num_time_series, times, features)`
                - `np.ndarray` of time with shape `(times)`
            - When `time_format` != TimeFormat.DATETIME or time is not included:
                - `np.ndarray` of shape `(num_time_series, times, features)`

        The `DataLoader` is configured with the following config attributes:

        | Dataset config                     | Description                                                                                                                               |
        | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
        | `test_batch_size`                  | Number of samples per batch. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.         |
        | `sliding_window_size`              | Available only for time-based datasets. Modifies the shape of the returned data.                                                          |
        | `sliding_window_prediction_size`   | Available only for time-based datasets. Modifies the shape of the returned data.                                                          |
        | `sliding_window_step`              | Available only for time-based datasets. Number of times to move by after each window.                                                     |
        | `test_workers`                     | Specifies the number of workers to use for loading test data. Applied when `workers` = "config".                                          |

        Parameters:
            workers: The number of workers to use for loading test data. `Default: "config"`  
            ts_id: Specifies time series to take. If None returns all time series as normal. `Default: "None"`

        Returns:
            An iterable `DataLoader` containing data from test set.        
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access test_dataloader.")

        if not self.dataset_config.has_test():
            raise ValueError("Dataloader for test set is not available in the dataset configuration.")

        assert self.test_dataset is not None, "The test_dataset must be initialized before accessing data from test set."

        default_kwargs = {'take_all': False, "cache_loader": True}
        kwargs = {**default_kwargs, **kwargs}

        if ts_id is not None:

            if ts_id == self.dataset_config.used_singular_test_time_series and self.test_dataloader is not None:
                self.logger.debug("Returning cached test_dataloader.")
                return self.test_dataloader

            dataset = self._get_singular_time_series_dataset(self.test_dataset, ts_id)
            self.dataset_config.used_singular_test_time_series = ts_id
            if self.test_dataloader:
                del self.test_dataloader
                self.test_dataloader = None
                self.logger.info("Destroyed previous cached test_dataloader.")

            self.dataset_config.used_test_workers = 0
            self.test_dataloader = self._get_dataloader(dataset, 0, False, self.dataset_config.test_batch_size)
            self.logger.info("Created new cached test_dataloader.")
            return self.test_dataloader
        elif self.dataset_config.used_singular_test_time_series is not None and self.test_dataloader is not None:
            del self.test_dataloader
            self.test_dataloader = None
            self.dataset_config.used_singular_test_time_series = None
            self.logger.info("Destroyed previous cached test_dataloader.")

        if workers == "config":
            workers = self.dataset_config.test_workers

        # If the dataloader is cached and number of used workers did not change, return the cached dataloader
        if self.test_dataloader and kwargs["cache_loader"] and workers == self.dataset_config.used_test_workers:
            self.logger.debug("Returning cached test_dataloader.")
            return self.test_dataloader

        # Update the used workers count
        self.dataset_config.used_test_workers = workers

        # If there's a previously cached dataloader, destroy it
        if self.test_dataloader:
            del self.test_dataloader
            self.test_dataloader = None
            self.logger.info("Destroyed previous cached test_dataloader.")

        # If caching is enabled, create a new cached dataloader
        if kwargs["cache_loader"]:
            self.test_dataloader = self._get_dataloader(self.test_dataset, workers, kwargs['take_all'], self.dataset_config.test_batch_size)
            self.logger.info("Created new cached test_dataloader.")
            return self.test_dataloader

        # If caching is disabled, create a new uncached dataloader
        self.logger.debug("Created new uncached test_dataloader.")
        return self._get_dataloader(self.test_dataset, workers, kwargs['take_all'], self.dataset_config.test_batch_size)

    def get_all_dataloader(self, ts_id: int | None = None, workers: int | Literal["config"] = "config", **kwargs) -> DataLoader:
        """
        Returns a PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for all set.

        The `DataLoader` is created on the first call and cached for subsequent use. <br/>
        The cached dataloader is cleared when either [`get_all_df`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_all_df] or [`get_all_numpy`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_all_numpy] is called.

        The structure of the returned batch depends on the `time_format` and whether `sliding_window_size` is used:

        - When `sliding_window_size` is used:
            - With `time_format` == TimeFormat.DATETIME and included time:
                - `np.ndarray` of shape `(num_time_series, times - 1, features)`
                - `np.ndarray` of shape `(num_time_series, 1, features)`
                - `np.ndarray` of times with shape `(times - 1)`
                - `np.ndarray` of time with shape `(1)`
            - When `time_format` != TimeFormat.DATETIME or time is not included:
                - `np.ndarray` of shape `(num_time_series, times - 1, features)`
                - `np.ndarray` of shape `(num_time_series, 1, features)`
        - When `sliding_window_size` is not used:
            - With `time_format` == TimeFormat.DATETIME and included time:
                - `np.ndarray` of shape `(num_time_series, times, features)`
                - `np.ndarray` of time with shape `(times)`
            - When `time_format` != TimeFormat.DATETIME or time is not included:
                - `np.ndarray` of shape `(num_time_series, times, features)`

        The `DataLoader` is configured with the following config attributes:

        | Dataset config                    | Description                                                                                                                               |
        | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
        | `all_batch_size`                  | Number of samples per batch. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.         |
        | `sliding_window_size`             | Available only for time-based datasets. Modifies the shape of the returned data.                                                          |
        | `sliding_window_prediction_size`  | Available only for time-based datasets. Modifies the shape of the returned data.                                                          |
        | `sliding_window_step`             | Available only for time-based datasets. Number of times to move by after each window.                                                     |
        | `all_workers`                     | Specifies the number of workers to use for loading all data. Applied when `workers` = "config".                                           |

        Parameters:
            workers: The number of workers to use for loading all data. `Default: "config"`  
            ts_id: Specifies time series to take. If None returns all time series as normal. `Default: "None"`

        Returns:
            An iterable `DataLoader` containing data from all set.       
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access all_dataloader.")

        if not self.dataset_config.has_all():
            raise ValueError("Dataloader for all set is not available in the dataset configuration.")

        assert self.all_dataset is not None, "The all_dataset must be initialized before accessing data from all set."

        default_kwargs = {'take_all': False, "cache_loader": True}
        kwargs = {**default_kwargs, **kwargs}

        if ts_id is not None:

            if ts_id == self.dataset_config.used_singular_all_time_series and self.all_dataloader is not None:
                self.logger.debug("Returning cached all_dataloader.")
                return self.all_dataloader

            dataset = self._get_singular_time_series_dataset(self.all_dataset, ts_id)
            self.dataset_config.used_singular_all_time_series = ts_id
            if self.all_dataloader:
                del self.all_dataloader
                self.all_dataloader = None
                self.logger.info("Destroyed previous cached all_dataloader.")

            self.dataset_config.used_all_workers = 0
            self.all_dataloader = self._get_dataloader(dataset, 0, False, self.dataset_config.all_batch_size)
            self.logger.info("Created new cached all_dataloader.")
            return self.all_dataloader
        elif self.dataset_config.used_singular_all_time_series is not None and self.all_dataloader is not None:
            del self.all_dataloader
            self.all_dataloader = None
            self.dataset_config.used_singular_all_time_series = None
            self.logger.info("Destroyed previous cached all_dataloader.")

        if workers == "config":
            workers = self.dataset_config.all_workers

        # If the dataloader is cached and number of used workers did not change, return the cached dataloader
        if self.all_dataloader and kwargs["cache_loader"] and workers == self.dataset_config.used_all_workers:
            self.logger.debug("Returning cached all_dataloader.")
            return self.all_dataloader

        # Update the used workers count
        self.dataset_config.used_all_workers = workers

        # If there's a previously cached dataloader, destroy it
        if self.all_dataloader:
            del self.all_dataloader
            self.all_dataloader = None
            self.logger.info("Destroyed previous cached all_dataloader.")

        # If caching is enabled, create a new cached dataloader
        if kwargs["cache_loader"]:
            self.all_dataloader = self._get_dataloader(self.all_dataset, workers, kwargs['take_all'], self.dataset_config.all_batch_size)
            self.logger.info("Created new cached all_dataloader.")
            return self.all_dataloader

        # If caching is disabled, create a new uncached dataloader
        self.logger.debug("Creating new uncached all_dataloader.")
        return self._get_dataloader(self.all_dataset, workers, kwargs['take_all'], self.dataset_config.all_batch_size)

    def get_train_df(self, workers: int | Literal["config"] = "config", as_single_dataframe: bool = True) -> pd.DataFrame:
        """
        Creates a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing all the data from training set grouped by time series.

        This method uses the `train_dataloader` with a batch size set to the total number of data in the training set. The cached `train_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire training set is loaded into memory, which may lead to high memory usage. If working with large training set, consider using `get_train_dataloader` instead to handle data in batches.

        Parameters:
            workers: The number of workers to use for loading train data. `Default: "config"`  
            as_single_dataframe: Whether to return a single dataframe with all time series combined, or to create separate dataframes for each time series. `Default: True` 

        Returns:
            A single Pandas DataFrame containing all data from training set, or a list of DataFrames (one per time series).
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access train_dataloader.")

        if not self.dataset_config.has_train():
            raise ValueError("Dataloader for training set is not available in the dataset configuration.")

        assert self.train_dataset is not None, "The train_dataset must be initialized before accessing data from training set."

        ts_ids, time_period = self.dataset_config._get_train()

        should_take_all = self.dataset_config.dataset_type != DatasetType.SERIES_BASED

        dataloader = self.get_train_dataloader(workers=workers, take_all=should_take_all, cache_loader=False)
        return self._get_df(dataloader, as_single_dataframe, ts_ids, time_period)

    def get_val_df(self, workers: int | Literal["config"] = "config", as_single_dataframe: bool = True) -> pd.DataFrame:
        """
        Create a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing all the data from validation set grouped by time series.

        This method uses the `val_dataloader` with a batch size set to the total number of data in the validation set. The cached `val_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire validation set is loaded into memory, which may lead to high memory usage. If working with large validation set, consider using `get_val_dataloader` instead to handle data in batches.

        Parameters:
            workers: The number of workers to use for loading validation data. `Default: "config"`  
            as_single_dataframe: Whether to return a single dataframe with all time series combined, or to create separate dataframes for each time series. `Default: True` 

        Returns:
            A single Pandas DataFrame containing all data from validation set, or a list of DataFrames (one per time series).
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access val_dataloader.")

        if not self.dataset_config.has_val():
            raise ValueError("Dataloader for validation set is not available in the dataset configuration.")

        assert self.val_dataset is not None, "The val_dataset must be initialized before accessing data from validation set."

        ts_ids, time_period = self.dataset_config._get_val()

        should_take_all = self.dataset_config.dataset_type != DatasetType.SERIES_BASED

        dataloader = self.get_val_dataloader(workers=workers, take_all=should_take_all, cache_loader=False)
        return self._get_df(dataloader, as_single_dataframe, ts_ids, time_period)

    def get_test_df(self, workers: int | Literal["config"] = "config", as_single_dataframe: bool = True) -> pd.DataFrame:
        """
        Creates a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing all the data from test set grouped by time series.

        This method uses the `test_dataloader` with a batch size set to the total number of data in the test set. The cached `test_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire test set is loaded into memory, which may lead to high memory usage. If working with large test set, consider using `get_test_dataloader` instead to handle data in batches.

        Parameters:
            workers: The number of workers to use for loading test data. `Default: "config"`  
            as_single_dataframe: Whether to return a single dataframe with all time series combined, or to create separate dataframes for each time series. `Default: True` 

        Returns:
            A single Pandas DataFrame containing all data from test set, or a list of DataFrames (one per time series).
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access test_dataloader.")

        if not self.dataset_config.has_test():
            raise ValueError("Dataloader for test set is not available in the dataset configuration.")

        assert self.test_dataset is not None, "The test_dataset must be initialized before accessing data from test set."

        ts_ids, time_period = self.dataset_config._get_test()

        should_take_all = self.dataset_config.dataset_type != DatasetType.SERIES_BASED

        dataloader = self.get_test_dataloader(workers=workers, take_all=should_take_all, cache_loader=False)
        return self._get_df(dataloader, as_single_dataframe, ts_ids, time_period)

    def get_all_df(self, workers: int | Literal["config"] = "config", as_single_dataframe: bool = True) -> pd.DataFrame:
        """
        Creates a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing all the data from all set grouped by time series.

        This method uses the `all_dataloader` with a batch size set to the total number of data in the all set. The cached `all_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire all set is loaded into memory, which may lead to high memory usage. If working with large all set, consider using `get_all_dataloader` instead to handle data in batches.

        Parameters:
            workers: The number of workers to use for loading all data. `Default: "config"`  
            as_single_dataframe: Whether to return a single dataframe with all time series combined, or to create separate dataframes for each time series. `Default: True` 

        Returns:
            A single Pandas DataFrame containing all data from all set, or a list of DataFrames (one per time series).
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access all_dataloader.")

        if not self.dataset_config.has_all():
            raise ValueError("Dataloader for all set is not available in the dataset configuration.")

        assert self.all_dataset is not None, "The all_dataset must be initialized before accessing data from all set."

        ts_ids, time_period = self.dataset_config._get_all()

        should_take_all = self.dataset_config.dataset_type != DatasetType.SERIES_BASED

        dataloader = self.get_all_dataloader(workers=workers, take_all=should_take_all, cache_loader=False)
        return self._get_df(dataloader, as_single_dataframe, ts_ids, time_period)

    def get_train_numpy(self, workers: int | Literal["config"] = "config") -> np.ndarray:
        """
        Creates a NumPy array containing all the data from training set grouped by time series, with the shape `(num_time_series, num_times, num_features)`.

        This method uses the `train_dataloader` with a batch size set to the total number of data in the training set. The cached `train_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire training set is loaded into memory, which may lead to high memory usage. If working with large training set, consider using `get_train_dataloader` instead to handle data in batches.        

        Parameters:
            workers: The number of workers to use for loading train data. `Default: "config"`  

        Returns:
            A NumPy array containing all the data in training set with the shape `(num_time_series, num_times, num_features)`.
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access train_dataloader.")

        if not self.dataset_config.has_train():
            raise ValueError("Dataloader for training set is not available in the dataset configuration.")

        assert self.train_dataset is not None, "The train_dataset must be initialized before accessing data from training set."

        ts_ids, time_period = self.dataset_config._get_train()

        should_take_all = self.dataset_config.dataset_type != DatasetType.SERIES_BASED

        dataloader = self.get_train_dataloader(workers=workers, take_all=should_take_all, cache_loader=False)
        return self._get_numpy(dataloader, ts_ids, time_period)

    def get_val_numpy(self, workers: int | Literal["config"] = "config") -> np.ndarray:
        """
        Creates a NumPy array containing all the data from validation set grouped by time series, with the shape `(num_time_series, num_times, num_features)`.

        This method uses the `val_dataloader` with a batch size set to the total number of data in the validation set. The cached `val_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire validation set is loaded into memory, which may lead to high memory usage. If working with large validation set, consider using `get_val_dataloader` instead to handle data in batches.        

        Parameters:
            workers: The number of workers to use for loading validation data. `Default: "config"`  

        Returns:
            A NumPy array containing all the data in validation set with the shape `(num_time_series, num_times, num_features)`.
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access val_dataloader.")

        if not self.dataset_config.has_val():
            raise ValueError("Dataloader for validation set is not available in the dataset configuration.")

        assert self.val_dataset is not None, "The val_dataset must be initialized before accessing data from validation set."

        ts_ids, time_period = self.dataset_config._get_val()

        should_take_all = self.dataset_config.dataset_type != DatasetType.SERIES_BASED

        dataloader = self.get_val_dataloader(workers=workers, take_all=should_take_all, cache_loader=False)
        return self._get_numpy(dataloader, ts_ids, time_period)

    def get_test_numpy(self, workers: int | Literal["config"] = "config") -> np.ndarray:
        """
        Creates a NumPy array containing all the data from test set grouped by time series, with the shape `(num_time_series, num_times, num_features)`.

        This method uses the `test_dataloader` with a batch size set to the total number of data in the test set. The cached `test_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire test set is loaded into memory, which may lead to high memory usage. If working with large test set, consider using `get_test_dataloader` instead to handle data in batches.        

        Parameters:
            workers: The number of workers to use for loading test data. `Default: "config"`  

        Returns:
            A NumPy array containing all the data in test set with the shape `(num_time_series, num_times, num_features)`.
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access test_dataloader.")

        if not self.dataset_config.has_test():
            raise ValueError("Dataloader for test set is not available in the dataset configuration.")

        assert self.test_dataset is not None, "The test_dataset must be initialized before accessing data from test set."

        ts_ids, time_period = self.dataset_config._get_test()

        should_take_all = self.dataset_config.dataset_type != DatasetType.SERIES_BASED

        dataloader = self.get_test_dataloader(workers=workers, take_all=should_take_all, cache_loader=False)
        return self._get_numpy(dataloader, ts_ids, time_period)

    def get_all_numpy(self, workers: int | Literal["config"] = "config") -> np.ndarray:
        """
        Creates a NumPy array containing all the data from all set grouped by time series, with the shape `(num_time_series, num_times, num_features)`.

        This method uses the `all_dataloader` with a batch size set to the total number of data in the all set. The cached `all_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire all set is loaded into memory, which may lead to high memory usage. If working with large all set, consider using `get_all_dataloader` instead to handle data in batches.        

        Parameters:
            workers: The number of workers to use for loading all data. `Default: "config"`  

        Returns:
            A NumPy array containing all the data in all set with the shape `(num_time_series, num_times, num_features)`.
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access all_dataloader.")

        if not self.dataset_config.has_all():
            raise ValueError("Dataloader for all set is not available in the dataset configuration.")

        assert self.all_dataset is not None, "The all_dataset must be initialized before accessing data from all set."

        ts_ids, time_period = self.dataset_config._get_all()

        should_take_all = self.dataset_config.dataset_type != DatasetType.SERIES_BASED

        dataloader = self.get_all_dataloader(workers=workers, take_all=should_take_all, cache_loader=False)
        return self._get_numpy(dataloader, ts_ids, time_period)

    def update_dataset_config_and_initialize(self,
                                             default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None | Literal["config"] = "config",
                                             sliding_window_size: int | None | Literal["config"] = "config",
                                             sliding_window_prediction_size: int | None | Literal["config"] = "config",
                                             sliding_window_step: int | Literal["config"] = "config",
                                             set_shared_size: float | int | Literal["config"] = "config",
                                             train_batch_size: int | Literal["config"] = "config",
                                             val_batch_size: int | Literal["config"] = "config",
                                             test_batch_size: int | Literal["config"] = "config",
                                             all_batch_size: int | Literal["config"] = "config",
                                             fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None | Literal["config"] = "config",
                                             transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None | Literal["config"] = "config",
                                             handle_anomalies_with: type | AnomalyHandlerType | Literal["z-score", "interquartile_range"] | None | Literal["config"] = "config",
                                             create_transformer_per_time_series: bool | Literal["config"] = "config",
                                             partial_fit_initialized_transformers: bool | Literal["config"] = "config",
                                             train_workers: int | Literal["config"] = "config",
                                             val_workers: int | Literal["config"] = "config",
                                             test_workers: int | Literal["config"] = "config",
                                             all_workers: int | Literal["config"] = "config",
                                             init_workers: int | Literal["config"] = "config",
                                             workers: int | Literal["config"] = "config",
                                             display_config_details: bool = False):
        """Used for updating selected configurations set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Can affect following configuration. 

        | Dataset config                          | Description                                                                                                                                     |
        | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
        | `default_values`                        | Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature.                        |  
        | `sliding_window_size`                   | Number of times in one window. Impacts dataloader behavior. Refer to relevant config for details.                                               |
        | `sliding_window_prediction_size`        | Number of times to predict from sliding_window_size. Refer to relevant config for details.                                                      |
        | `sliding_window_step`                   | Number of times to move by after each window. Refer to relevant config for details.                                                             |
        | `set_shared_size`                       | How much times should time periods share. Order of sharing is training set < validation set < test set. Refer to relevant config for details.   |           
        | `train_batch_size`                      | Number of samples per batch for train set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details. |
        | `val_batch_size`                        | Number of samples per batch for val set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.   |
        | `test_batch_size`                       | Number of samples per batch for test set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.  |
        | `all_batch_size`                        | Number of samples per batch for all set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.   |                   
        | `fill_missing_with`                     | Defines how to fill missing values in the dataset.                                                                                              |                
        | `transform_with`                        | Defines the transformer to transform the dataset.                                                                                               | 
        | `handle_anomalies_with`                 | Defines the anomaly handler to handle anomalies in the dataset.                                                                                 |            
        | `create_transformer_per_time_series`    | If `True`, a separate transformer is created for each time series. Not used when using already initialized transformers.                        |   
        | `partial_fit_initialized_transformers`  | If `True`, partial fitting on train set is performed when using initiliazed transformers.                                                       |   
        | `train_workers`                         | Number of workers for loading training data.                                                                                                    |
        | `val_workers`                           | Number of workers for loading validation data.                                                                                                  |
        | `test_workers`                          | Number of workers for loading test data.                                                                                                        |
        | `all_workers`                           | Number of workers for loading all data.                                                                                                         |     
        | `init_workers`                          | Number of workers for dataset configuration.                                                                                                    |                        

        Parameters:
            default_values: Default values for missing data, applied before fillers. `Defaults: config`.  
            sliding_window_size: Number of times in one window. `Defaults: config`.
            sliding_window_prediction_size: Number of times to predict from sliding_window_size. `Defaults: config`.
            sliding_window_step: Number of times to move by after each window. `Defaults: config`.
            set_shared_size: How much times should time periods share. `Defaults: config`.            
            train_batch_size: Number of samples per batch for train set. `Defaults: config`.
            val_batch_size: Number of samples per batch for val set. `Defaults: config`.
            test_batch_size: Number of samples per batch for test set. `Defaults: config`.
            all_batch_size: Number of samples per batch for all set. `Defaults: config`.                    
            fill_missing_with: Defines how to fill missing values in the dataset. `Defaults: config`. 
            transform_with: Defines the transformer to transform the dataset. `Defaults: config`.  
            handle_anomalies_with: Defines the anomaly handler to handle anomalies in the dataset. `Defaults: config`.  
            create_transformer_per_time_series: If `True`, a separate transformer is created for each time series. Not used when using already initialized transformers. `Defaults: config`.  
            partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers. `Defaults: config`.    
            train_workers: Number of workers for loading training data. `Defaults: config`.
            val_workers: Number of workers for loading validation data. `Defaults: config`.
            test_workers: Number of workers for loading test data. `Defaults: config`.
            all_workers: Number of workers for loading all data.  `Defaults: config`.
            init_workers: Number of workers for dataset configuration. `Defaults: config`.                          
            workers: How many workers to use when updating configuration. `Defaults: config`.  
            display_config_details: Whether config details should be displayed after configuration. `Defaults: False`. 
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating dataset configuration.")

        requires_init = False

        if default_values == "config":
            default_values = self._export_config_copy.default_values
        else:
            requires_init = True

        if isinstance(self.dataset_config, TimeBasedHandler):
            if sliding_window_size == "config":
                sliding_window_size = self.dataset_config.sliding_window_size
            if sliding_window_prediction_size == "config":
                sliding_window_prediction_size = self.dataset_config.sliding_window_prediction_size
            if sliding_window_step == "config":
                sliding_window_step = self.dataset_config.sliding_window_step
            if set_shared_size == "config":
                set_shared_size = self.dataset_config.set_shared_size
            else:
                requires_init = True

        if train_batch_size == "config":
            train_batch_size = self.dataset_config.train_batch_size
        if val_batch_size == "config":
            val_batch_size = self.dataset_config.val_batch_size
        if test_batch_size == "config":
            test_batch_size = self.dataset_config.test_batch_size
        if all_batch_size == "config":
            all_batch_size = self.dataset_config.all_batch_size

        if fill_missing_with == "config":
            fill_missing_with = self._export_config_copy.filler_factory.filler_type
        else:
            requires_init = True

        if create_transformer_per_time_series == "config":
            create_transformer_per_time_series = self._export_config_copy.create_transformer_per_time_series
        else:
            requires_init = True

        if partial_fit_initialized_transformers == "config":
            partial_fit_initialized_transformers = self._export_config_copy.partial_fit_initialized_transformers
        else:
            requires_init = True

        if transform_with == "config":
            if self._export_config_copy.transformer_factory.has_already_initialized:
                transform_with = self._export_config_copy.transformer_factory.initialized_transformers
            else:
                transform_with = self._export_config_copy.transformer_factory.transformer_type
        else:
            requires_init = True

        if handle_anomalies_with == "config":
            handle_anomalies_with = self._export_config_copy.anomaly_handler_factory.anomaly_handler_type
        else:
            requires_init = True

        if train_workers == "config":
            train_workers = self.dataset_config.train_workers
        if val_workers == "config":
            val_workers = self.dataset_config.val_workers
        if test_workers == "config":
            test_workers = self.dataset_config.test_workers
        if all_workers == "config":
            all_workers = self.dataset_config.all_workers
        if init_workers == "config":
            init_workers = self.dataset_config.init_workers

        original_config = deepcopy(self.dataset_config)
        original_export_config = deepcopy(self._export_config_copy)
        try:
            if requires_init:
                self.logger.info("Re-initialization is required.")
                self._export_config_copy.default_values = default_values
                if isinstance(self.dataset_config, TimeBasedHandler):
                    self._export_config_copy.sliding_window_size = sliding_window_size
                    self._export_config_copy.sliding_window_prediction_size = sliding_window_prediction_size
                    self._export_config_copy.sliding_window_step = sliding_window_step
                    self._export_config_copy.set_shared_size = set_shared_size
                self._export_config_copy.train_batch_size = train_batch_size
                self._export_config_copy.val_batch_size = val_batch_size
                self._export_config_copy.test_batch_size = test_batch_size
                self._export_config_copy.all_batch_size = all_batch_size
                self._export_config_copy.filler_factory = filler_factories.get_filler_factory(fill_missing_with)
                self._export_config_copy.transformer_factory = transformer_factories.get_transformer_factory(transform_with, create_transformer_per_time_series, partial_fit_initialized_transformers)
                self._export_config_copy.anomaly_handler_factory = anomaly_handler_factories.get_anomaly_handler_factory(handle_anomalies_with)
                self._export_config_copy.partial_fit_initialized_transformers = partial_fit_initialized_transformers
                self._export_config_copy.create_transformer_per_time_series = create_transformer_per_time_series
                self._export_config_copy.train_workers = train_workers
                self._export_config_copy.val_workers = val_workers
                self._export_config_copy.test_workers = test_workers
                self._export_config_copy.all_workers = all_workers
                self._export_config_copy.init_workers = init_workers
                self._export_config_copy._validate_construction()
                self.set_dataset_config_and_initialize(self._export_config_copy, False, workers)
            else:
                self.logger.info("Re-initialization is not needed.")
                self.dataset_config._update_batch_sizes(train_batch_size, val_batch_size, test_batch_size, all_batch_size)
                self.dataset_config._update_workers(train_workers, val_workers, test_workers, all_workers, init_workers)

                if isinstance(self.dataset_config, TimeBasedHandler):
                    self.dataset_config._update_sliding_window(sliding_window_size, sliding_window_prediction_size, sliding_window_step, set_shared_size, self.metadata.time_indices)

                if self.train_dataloader is not None:
                    del self.train_dataloader
                    self.train_dataloader = None
                    self.logger.info("Destroyed cached train_dataloader.")

                if self.val_dataloader is not None:
                    del self.val_dataloader
                    self.val_dataloader = None
                    self.logger.info("Destroyed cached val_dataloader.")

                if self.test_dataloader is not None:
                    del self.test_dataloader
                    self.test_dataloader = None
                    self.logger.info("Destroyed cached test_dataloader.")

                if self.all_dataloader is not None:
                    del self.all_dataloader
                    self.all_dataloader = None
                    self.logger.info("Destroyed cached all_dataloader.")
        except Exception:
            self.dataset_config = original_config
            self._export_config_copy = original_export_config
            self.logger.error("Error occured, reverting changes.")
            raise

        self._update_config_imported_status(None)
        self._update_export_config_copy()

        self.logger.info("Configuration has been changed successfuly.")

        if display_config_details:
            self.display_config()

    def apply_filler(self, fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None, workers: int | Literal["config"] = "config") -> None:
        """Used for updating filler set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                     | Description                                        |
        | ---------------------------------- | -------------------------------------------------- |
        | `fill_missing_with`                | Defines how to fill missing values in the dataset. |     

        Parameters:
            fill_missing_with: Defines how to fill missing values in the dataset. `Defaults: config`.  
            workers: How many workers to use when setting new filler. `Defaults: config`.      
        """
        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating filler.")

        self.update_dataset_config_and_initialize(fill_missing_with=fill_missing_with, workers=workers)
        self.logger.info("Filler has been changed successfuly.")

    def apply_anomaly_handler(self, handle_anomalies_with: type | AnomalyHandlerType | Literal["z-score", "interquartile_range"] | None | Literal["config"], workers: int | Literal["config"] = "config") -> None:
        """Used for updating anomaly handler set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                     | Description                                                            |
        | ---------------------------------- | ---------------------------------------------------------------------- |
        | `handle_anomalies_with`            | Defines the anomaly handler to handle anomalies in the dataset.        |     

        Parameters:
            handle_anomalies_with: Defines the anomaly handler to handle anomalies in the dataset. `Defaults: config`.  
            workers: How many workers to use when setting new filler. `Defaults: config`.      
        """
        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating anomaly handler.")

        self.update_dataset_config_and_initialize(handle_anomalies_with=handle_anomalies_with, workers=workers)
        self.logger.info("Anomaly handler has been changed successfuly.")

    def apply_transformer(self, transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None | Literal["config"] = "config",
                          create_transformer_per_time_series: bool | Literal["config"] = "config", partial_fit_initialized_transformers: bool | Literal["config"] = "config", workers: int | Literal["config"] = "config") -> None:
        """Used for updating transformer and relevenat configurations set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                          | Description                                                                                                              |
        | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
        | `transform_with`                        | Defines the transformer to transform the dataset.                                                                        |     
        | `create_transformer_per_time_series`    | If `True`, a separate transformer is created for each time series. Not used when using already initialized transformers. |   
        | `partial_fit_initialized_transformers`  | If `True`, partial fitting on train set is performed when using initiliazed transformers.                                |    

        Parameters:
            transform_with: Defines the transformer to transform the dataset. `Defaults: config`.  
            create_transformer_per_time_series: If `True`, a separate transformer is created for each time series. Not used when using already initialized transformers. `Defaults: config`.  
            partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers. `Defaults: config`.  
            workers: How many workers to use when setting new transformer. `Defaults: config`.      
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating transformer values.")

        self.update_dataset_config_and_initialize(transform_with=transform_with, create_transformer_per_time_series=create_transformer_per_time_series, partial_fit_initialized_transformers=partial_fit_initialized_transformers, workers=workers)
        self.logger.info("Transformer configuration has been changed successfuly.")

    def set_default_values(self, default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None, workers: int | Literal["config"] = "config") -> None:
        """Used for updating default values set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                     | Description                                                                                                              |
        | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
        | `default_values`                   | Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature. |     

        Parameters:
            default_values: Default values for missing data, applied before fillers. `Defaults: config`.  
            workers: How many workers to use when setting new default values. `Defaults: config`.      
        """
        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating default values.")

        self.update_dataset_config_and_initialize(default_values=default_values, workers=workers)
        self.logger.info("Default values has been changed successfuly.")

    def set_workers(self, train_workers: int | Literal["config"] = "config", val_workers: int | Literal["config"] = "config",
                    test_workers: int | Literal["config"] = "config", all_workers: int | Literal["config"] = "config", init_workers: int | Literal["config"] = "config") -> None:
        """Used for updating workers set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                 | Description                                    |
        | ------------------------------ | ---------------------------------------------- |
        | `train_workers`                | Number of workers for loading training data.   |
        | `val_workers`                  | Number of workers for loading validation data. |
        | `test_workers`                 | Number of workers for loading test data.       |
        | `all_workers`                  | Number of workers for loading all data.        |     
        | `init_workers`                 | Number of workers for dataset configuration.   |      

        Parameters:
            train_workers: Number of workers for loading training data. `Defaults: config`.
            val_workers: Number of workers for loading validation data. `Defaults: config`.
            test_workers: Number of workers for loading test data. `Defaults: config`.
            all_workers: Number of workers for loading all data.  `Defaults: config`.
            init_workers: Number of workers for dataset configuration. `Defaults: config`.            
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating workers.")

        self.update_dataset_config_and_initialize(train_workers=train_workers, val_workers=val_workers, test_workers=test_workers, all_workers=all_workers, init_workers=init_workers, workers="config")
        self.logger.info("Workers has been changed successfuly.")

    def set_batch_sizes(self, train_batch_size: int | Literal["config"] = "config", val_batch_size: int | Literal["config"] = "config",
                        test_batch_size: int | Literal["config"] = "config", all_batch_size: int | Literal["config"] = "config") -> None:
        """Used for updating batch sizes set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                    | Description                                                                                                                                     |
        | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
        | `train_batch_size`                | Number of samples per batch for train set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details. |
        | `val_batch_size`                  | Number of samples per batch for val set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.   |
        | `test_batch_size`                 | Number of samples per batch for test set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.  |
        | `all_batch_size`                  | Number of samples per batch for all set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.   |        

        Parameters:
            train_batch_size: Number of samples per batch for train set. `Defaults: config`.
            val_batch_size: Number of samples per batch for val set. `Defaults: config`.
            test_batch_size: Number of samples per batch for test set. `Defaults: config`.
            all_batch_size: Number of samples per batch for all set. `Defaults: config`.
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating batch sizes.")

        self.update_dataset_config_and_initialize(train_batch_size=train_batch_size, val_batch_size=val_batch_size, test_batch_size=test_batch_size, all_batch_size=all_batch_size, workers="config")
        self.logger.info("Batch sizes has been changed successfuly.")

    def display_dataset_details(self) -> None:
        """Display information about the contents of the dataset.  """

        to_display = f'''
Dataset details:

    {self.metadata.aggregation}
        Time indices: {range(self.metadata.time_indices[ID_TIME_COLUMN_NAME][0], self.metadata.time_indices[ID_TIME_COLUMN_NAME][-1])}
        Datetime: {(datetime.fromtimestamp(self.metadata.time_indices['time'][0], tz=timezone.utc), datetime.fromtimestamp(self.metadata.time_indices['time'][-1], timezone.utc))}

    {self.metadata.source_type}
        Time series indices: {get_abbreviated_list_string(self.metadata.ts_indices[self.metadata.ts_id_name])}; use 'get_available_ts_indices' for full list
        Features with default values: {self.metadata.default_values}
        
        Additional data: {list(self.metadata.additional_data.keys())}
        '''

        print(to_display)

    def display_config(self) -> None:
        """Displays the values of the initialized configuration. """
        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before displaying config.")

        print(self.dataset_config)

    def get_feature_names(self) -> list[str]:
        """Returns a list of all available feature names in the dataset. """

        return list(self.metadata.features.keys())

    @abstractmethod
    def get_data_about_set(self, about: SplitType | Literal["train", "val", "test", "all"]) -> dict:
        """
        Retrieves data related to the specified set.

        Parameters:
            about: Specifies the set to retrieve data about.

        Returns:
            A dictionary containing the requested data for the set.
        """
        ...

    def get_available_ts_indices(self) -> np.ndarray:
        """Returns the available time series indices in this dataset. """
        return self.metadata.ts_indices

    def get_additional_data(self, data_name: str) -> pd.DataFrame:
        """Create a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) of additional data of `data_name`.

        Parameters:
            data_name: Name of additional data to return.

        Returns:
            Dataframe of additional data of `data_name`.
        """

        if data_name not in self.metadata.additional_data:
            self.logger.error("%s is not available for this dataset.", data_name)
            raise ValueError(f"{data_name} is not available for this dataset.", f"Possible options are: {self.metadata.additional_data}")

        data = get_additional_data(self.metadata.dataset_path, data_name)
        data_df = pd.DataFrame(data)

        for column, column_type in self.metadata.additional_data[data_name]:
            if column_type == datetime:
                data_df[column] = data_df[column].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
            else:
                data_df[column] = data_df[column].astype(column_type)

        return data_df

    def plot(self, ts_id: int, plot_type: Literal["scatter", "line"], features: list[str] | str | Literal["config"] = "config", feature_per_plot: bool = True,
             time_format: TimeFormat | Literal["config", "id_time", "datetime", "unix_time", "shifted_unix_time"] = "config", is_interactive: bool = True) -> None:
        """
        Displays a graph for the selected `ts_id` and its `features`.

        The plotting is done using the [`Plotly`](https://plotly.com/python/) library, which provides interactive graphs.

        Parameters:
            ts_id: The ID of the time series to display.
            plot_type: The type of graph to plot.
            features: The features to display in the plot. `Defaults: "config"`.
            feature_per_plot: Whether each feature should be displayed in a separate plot or combined into one. `Defaults: True`.
            time_format: The time format to use for the x-axis. `Defaults: "config"`.
            is_interactive: Whether the plot should be interactive (e.g., zoom, hover). `Defaults: True`.
        """

        if time_format == "config":

            if self.dataset_config is None or not self.dataset_config.is_initialized:
                raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to plot.")

            time_format = self.dataset_config.time_format
            self.logger.debug("Using time format from dataset configuration: %s", time_format)
        else:
            time_format = TimeFormat(time_format)
            self.logger.debug("Using specified time format: %s", time_format)

        time_series, times, features = self.__get_data_for_plot(ts_id, features, time_format)
        self.logger.debug("Received data for plotting. Time series, times, and features are ready.")

        plots = []

        if feature_per_plot:
            self.logger.debug("Creating individual plots for each feature.")
            fig = make_subplots(rows=len(features), cols=1, shared_xaxes=False, x_title=time_format.value)

            for i, feature in enumerate(features):
                if plot_type == "scatter":
                    plot = go.Scatter(x=times, y=time_series[:, i], mode="markers", name=feature, legendgroup=feature)
                    self.logger.debug("Creating scatter plot for feature: %s", feature)
                elif plot_type == "line":
                    plot = go.Scatter(x=times, y=time_series[:, i], mode="lines", name=feature)
                    self.logger.debug("Creating line plot for feature: %s", feature)
                else:
                    raise ValueError("Invalid plot type.")

                fig.add_traces(plot, rows=i + 1, cols=1)

            fig.update_layout(height=200 + 120 * len(features), width=2000, autosize=len(features) == 1, showlegend=True)
            self.logger.debug("Created subplots for features: %s.", features)
        else:
            self.logger.debug("Creating a combined plot for all features.")
            for i, feature in enumerate(features):
                if plot_type == "scatter":
                    plot = go.Scatter(x=times, y=time_series[:, i], mode="markers", name=feature)
                    self.logger.debug("Creating scatter plot for feature: %s", feature)
                elif plot_type == "line":
                    plot = go.Scatter(x=times, y=time_series[:, i], mode="lines", name=feature)
                    self.logger.debug("Creating line plot for feature: %s", feature)
                else:
                    raise ValueError("Invalid plot type.")
                plots.append(plot)

            fig = go.Figure(data=plots)
            fig.update_layout(xaxis_title=time_format.value, showlegend=True, height=200 + 120 * 2)
            self.logger.debug("Created combined plot for features: %s.", features)

        if not is_interactive:
            self.logger.debug("Disabling interactivity for the plot.")
            fig.update_layout(updatemenus=[], dragmode=False, hovermode=False)

        self.logger.debug("Displaying the plot.")
        fig.show()

    def add_annotation(self, annotation: str, annotation_group: str, ts_id: int | None, id_time: int | None, enforce_ids: bool = True) -> None:
        """ 
        Adds an annotation to the specified `annotation_group`.

        - If the provided `annotation_group` does not exist, it will be created.
        - At least one of `ts_id` or `id_time` must be provided to associate the annotation with time series or/and time point.

        Parameters:
            annotation: The annotation to be added.
            annotation_group: The group to which the annotation should be added.
            ts_id: The time series ID to which the annotation should be added.
            id_time: The time ID to which the annotation should be added.
            enforce_ids: Flag indicating whether the `ts_id` and `id_time` must belong to this dataset. `Default: True`  
        """

        if enforce_ids:
            self._validate_annotation_ids(ts_id, id_time)
        self.annotations.add_annotation(annotation, annotation_group, ts_id, id_time)

        if ts_id is not None and id_time is not None:
            self._update_annotations_imported_status(AnnotationType.BOTH, None)
        elif ts_id is not None and id_time is None:
            self._update_annotations_imported_status(AnnotationType.TS_ID, None)
        elif ts_id is None and id_time is not None:
            self._update_annotations_imported_status(AnnotationType.ID_TIME, None)

    def remove_annotation(self, annotation_group: str, ts_id: int | None, id_time: int | None) -> None:
        """  
        Removes an annotation from the specified `annotation_group`.

        - At least one of `ts_id` or `id_time` must be provided to associate the annotation with time series or/and time point.

        Parameters:
            annotation_group: The annotation group from which the annotation should be removed.
            ts_id: The time series ID from which the annotation should be removed.
            id_time: The time ID from which the annotation should be removed. 
        """

        self.annotations.remove_annotation(annotation_group, ts_id, id_time, False)

        if ts_id is not None and id_time is not None:
            self._update_annotations_imported_status(AnnotationType.BOTH, None)
        elif ts_id is not None and id_time is None:
            self._update_annotations_imported_status(AnnotationType.TS_ID, None)
        elif ts_id is None and id_time is not None:
            self._update_annotations_imported_status(AnnotationType.ID_TIME, None)

    def add_annotation_group(self, annotation_group: str, on: AnnotationType | Literal["id_time", "ts_id", "both"]):
        """ 
        Adds a new `annotation_group`.

        Parameters:
            annotation_group: The name of the annotation group to be added.
            on: Specifies which part of the data should be annotated. If set to `"both"`, annotations will be applied as if `id_time` and `ts_id` were both set.
        """
        on = AnnotationType(on)

        self.annotations.add_annotation_group(annotation_group, on, False)

        self._update_annotations_imported_status(on, None)

    def remove_annotation_group(self, annotation_group: str, on: AnnotationType | Literal["id_time", "ts_id", "both"]):
        """ 
        Removes the specified `annotation_group`.

        Parameters:
            annotation_group: The name of the annotation group to be removed.
            on: Specifies which part of the data the `annotation_group` should be removed from. If set to `"both"`, annotations will be applied as if `id_time` and `ts_id` were both set.        
        """
        on = AnnotationType(on)

        self.annotations.remove_annotation_group(annotation_group, on, False)

        self._update_annotations_imported_status(on, None)

    def get_annotations(self, on: AnnotationType | Literal["id_time", "ts_id", "both"]) -> pd.DataFrame:
        """ 
        Returns the annotations as a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

        Parameters:
            on: Specifies which annotations to return. If set to `"both"`, annotations will be applied as if `id_time` and `ts_id` were both set.         

        Returns:
            A Pandas DataFrame containing the selected annotations.      
        """
        on = AnnotationType(on)

        return self.annotations.get_annotations(on, self.metadata.ts_id_name)

    def import_annotations(self, identifier: str, enforce_ids: bool = True) -> None:
        """ 
        Imports annotations from a CSV file.

        First, it attempts to load the built-in annotations, if no built-in annotations with such an identifier exists, it attempts to load a custom annotations from the `"data_root"/tszoo/annotations/` directory.

        `data_root` is specified when the dataset is created.     

        Parameters:
            identifier: The name of the CSV file.     
            enforce_ids: Flag indicating whether the `ts_id` and `id_time` must belong to this dataset. `Default: True`                
        """

        annotations_file_path, is_built_in = get_annotations_path_and_whether_it_is_built_in(identifier, self.metadata.annotations_root, self.logger)

        if is_built_in:
            self.logger.info("Built-in annotations found: %s.", identifier)
            if not os.path.exists(annotations_file_path):
                self.logger.info("Downloading annotations with identifier: %s", identifier)
                annotations_url = f"{ANNOTATIONS_DOWNLOAD_BUCKET}&file={identifier}"  # probably will change annotations bucket... placeholder
                resumable_download(url=annotations_url, file_path=annotations_file_path, silent=False)

            self.logger.debug("Loading annotations from %s", annotations_file_path)
            temp_df = pd.read_csv(annotations_file_path)
            self.logger.debug("Created DataFrame from file: %s", annotations_file_path)
        else:
            self.logger.info("Custom annotations found: %s.", identifier)
            self.logger.debug("Loading annotations from %s", annotations_file_path)
            temp_df = pd.read_csv(annotations_file_path)
            self.logger.debug("Created DataFrame from file: %s", annotations_file_path)

        ts_id_index = None
        time_id_index = None
        on = None

        # Check the columns of the DataFrame to identify the type of annotation
        if self.metadata.ts_id_name in temp_df.columns and ID_TIME_COLUMN_NAME in temp_df.columns:
            self.annotations.clear_time_in_time_series()
            time_id_index = temp_df.columns.tolist().index(ID_TIME_COLUMN_NAME)
            ts_id_index = temp_df.columns.tolist().index(self.metadata.ts_id_name)
            on = AnnotationType.BOTH
            self.logger.info("Annotations detected as %s (both %s and id_time)", AnnotationType.BOTH, self.metadata.ts_id_name)

        elif self.metadata.ts_id_name in temp_df.columns:
            self.annotations.clear_time_series()
            ts_id_index = temp_df.columns.tolist().index(self.metadata.ts_id_name)
            on = AnnotationType.TS_ID
            self.logger.info("Annotations detected as %s (%s only)", AnnotationType.TS_ID, self.metadata.ts_id_name)

        elif ID_TIME_COLUMN_NAME in temp_df.columns:
            self.annotations.clear_time()
            time_id_index = temp_df.columns.tolist().index(ID_TIME_COLUMN_NAME)
            on = AnnotationType.ID_TIME
            self.logger.info("Annotations detected as %s (%s only)", AnnotationType.ID_TIME, ID_TIME_COLUMN_NAME)

        else:
            raise ValueError(f"Could not find {self.metadata.ts_id_name} and {ID_TIME_COLUMN_NAME} in the imported CSV.")

        # Process each row in the DataFrame and add annotations
        for row in temp_df.itertuples(False):
            for i, _ in enumerate(temp_df.columns):
                if i == time_id_index or i == ts_id_index:
                    continue

                ts_id = None
                if ts_id_index is not None:
                    ts_id = row[ts_id_index]

                id_time = None
                if time_id_index is not None:
                    id_time = row[time_id_index]

                self.add_annotation(row[i], temp_df.columns[i], ts_id, id_time, enforce_ids)

        self._update_annotations_imported_status(on, identifier)
        self.logger.info("Successfully imported annotations from %s", annotations_file_path)

    def import_config(self, identifier: str, display_config_details: bool = True, workers: int | Literal["config"] = "config") -> None:
        """ 
        Import the dataset_config from a pickle file and initializes the dataset. Config type must correspond to dataset type.

        First, it attempts to load the built-in config, if no built-in config with such an identifier exists, it attempts to load a custom config from the `"data_root"/tszoo/configs/` directory.

        `data_root` is specified when the dataset is created.       

        The following configuration attributes are used during initialization:

        | Dataset config                         | Description                                                                                    |
        | -------------------------------------- | ---------------------------------------------------------------------------------------------- |
        | `init_workers`                         | Specifies the number of workers to use for initialization. Applied when `workers` = "config".  |
        | `partial_fit_initialized_transformers` | Determines whether initialized transformers should be partially fitted on the training data.   |
        | `nan_threshold`                        | Filters out time series with missing values exceeding the specified threshold.                 |  

        Parameters:
            identifier: Name of the pickle file.
            display_config_details: Flag indicating whether to display the configuration values after initialization. `Default: True` 
            workers: The number of workers to use during initialization. `Default: "config"`  
        """

        # Load config
        config = load_config(identifier, self.metadata.configs_root, self.metadata.database_name, self.metadata.source_type, self.metadata.aggregation, self.logger)

        self.logger.info("Initializing dataset configuration with the imported config.")
        self.set_dataset_config_and_initialize(config, display_config_details, workers)

        self._update_config_imported_status(identifier)
        self.logger.info("Successfully used config with identifier %s", identifier)

    def save_annotations(self, identifier: str, on: AnnotationType | Literal["id_time", "ts_id", "both"], force_write: bool = False) -> None:
        """ 
        Saves the annotations as a CSV file.

        The file will be saved to a path determined by the `data_root` specified when the dataset was created.

        The annotations will be saved under the directory `data_root/tszoo/annotations/`.

        Parameters:
            identifier: The name of the CSV file.
            on: What annotation type should be saved. If set to `"both"`, annotations will be applied as if `id_time` and `ts_id` were both set.   
            force_write: If set to `True`, will overwrite any existing files with the same name. `Default: False`               
        """

        if exists_built_in_annotations(identifier):
            raise ValueError("Built-in annotations with this identifier already exists. Choose another identifier.")

        on = AnnotationType(on)

        temp_df = self.get_annotations(on)

        # Ensure the annotations root directory exists, creating it if necessary
        if not os.path.exists(self.metadata.annotations_root):
            os.makedirs(self.metadata.annotations_root)
            self.logger.info("Created annotations directory at %s", self.metadata.annotations_root)

        path = os.path.join(self.metadata.annotations_root, f"{identifier}.csv")

        if os.path.exists(path) and not force_write:
            raise ValueError(f"Annotations already exist at {path}. Set force_write=True to overwrite.")
        self.logger.debug("Annotations CSV file path: %s", path)

        temp_df.to_csv(path, index=False)

        self._update_annotations_imported_status(on, identifier)
        self.logger.info("Annotations successfully saved to %s", path)

    def save_config(self, identifier: str, create_with_details_file: bool = True, force_write: bool = False, **kwargs) -> None:
        """ 
        Saves the config as a pickle file.

        The file will be saved to a path determined by the `data_root` specified when the dataset was created. 
        The config will be saved under the directory `data_root/tszoo/configs/`.

        Parameters:
            identifier: The name of the pickle file.
            create_with_details_file: Whether to export the config along with a readable text file that provides details. `Defaults: True`. 
            force_write: If set to `True`, will overwrite any existing files with the same name. `Default: False`            
        """

        default_kwargs = {'hard_force': False}
        kwargs = {**default_kwargs, **kwargs}

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to save config.")

        if not kwargs["hard_force"] and exists_built_in_config(identifier):
            raise ValueError("Built-in config with this identifier already exists. Choose another identifier.")

        # Ensure the config directory exists
        if not os.path.exists(self.metadata.configs_root):
            os.makedirs(self.metadata.configs_root)
            self.logger.info("Created config directory at %s", self.metadata.configs_root)

        path_pickle = os.path.join(self.metadata.configs_root, f"{identifier}.pickle")
        path_details = os.path.join(self.metadata.configs_root, f"{identifier}.txt")

        if os.path.exists(path_pickle) and not force_write:
            raise ValueError(f"Config at path {path_pickle} already exists. Set force_write=True to overwrite.")
        self.logger.debug("Config pickle path: %s", path_pickle)

        if create_with_details_file:
            if os.path.exists(path_details) and not force_write:
                raise ValueError(f"Config details at path {path_details} already exists. Set force_write=True to overwrite.")
            self.logger.debug("Config details path: %s", path_details)

        if not self.dataset_config.filler_factory.creates_built_in:
            self.logger.warning("You are using a custom filler. Ensure the config is distributed with the source code of the filler.")

        if not self.dataset_config.anomaly_handler_factory.creates_built_in:
            self.logger.warning("You are using a custom anomaly handler. Ensure the config is distributed with the source code of the anomaly handler.")

        if not self.dataset_config.transformer_factory.creates_built_in:
            self.logger.warning("You are using a custom transformer. Ensure the config is distributed with the source code of the transformer.")

        pickle_dump(self._export_config_copy, path_pickle)
        self.logger.info("Config pickle saved to %s", path_pickle)

        if create_with_details_file:
            with open(path_details, "w", encoding="utf-8") as file:
                file.write(str(self.dataset_config))
            self.logger.info("Config details saved to %s", path_details)

        self._update_config_imported_status(identifier)
        self.dataset_config.export_update_needed = False
        self.logger.info("Config successfully saved")

    def save_benchmark(self, identifier: str, force_write: bool = False, **kwargs) -> None:
        """ 
        Saves the benchmark as a YAML file.

        The benchmark, along with any associated annotations and config files, will be saved in a path determined by the `data_root` specified when creating the dataset. 
        The default save path for benchmark is `"data_root/tszoo/benchmarks/"`.

        If you are using imported `annotations` or `config` (whether custom or built-in), their file names will be set in the `benchmark` file. 
        If new `annotations` or `config` are created during the process, their filenames will be derived from the provided `identifier` and set in the `benchmark` file.

        Parameters:
            identifier: The name of the YAML file.
            force_write: If set to `True`, will overwrite any existing files with the same name. `Default: False`            
        """

        default_kwargs = {'hard_force': False}
        kwargs = {**default_kwargs, **kwargs}

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to save benchmark.")

        if not kwargs["hard_force"] and exists_built_in_benchmark(identifier):
            raise ValueError("Built-in benchmark with this identifier already exists. Choose another identifier.")

        # Determine annotation names based on the available annotations and whether the annotations were imported
        if len(self.annotations.time_series_annotations) > 0:
            annotations_ts_name = self.imported_annotations_ts_identifier if self.imported_annotations_ts_identifier is not None else f"{identifier}_{AnnotationType.TS_ID.value}"
        else:
            annotations_ts_name = None

        if len(self.annotations.time_annotations) > 0:
            annotations_time_name = self.imported_annotations_time_identifier if self.imported_annotations_time_identifier is not None else f"{identifier}_{AnnotationType.ID_TIME.value}"
        else:
            annotations_time_name = None

        if len(self.annotations.time_in_series_annotations) > 0:
            annotations_both_name = self.imported_annotations_both_identifier if self.imported_annotations_both_identifier is not None else f"{identifier}_{AnnotationType.BOTH.value}"
        else:
            annotations_both_name = None

        # Use the imported identifier if available and update is not necessary, otherwise default to the current identifier
        config_name = self.dataset_config.import_identifier if (self.dataset_config.import_identifier is not None and not self.dataset_config.export_update_needed) else identifier

        export_benchmark = ExportBenchmark(self.metadata.database_name,
                                           self.metadata.source_type.value,
                                           self.metadata.aggregation.value,
                                           self.metadata.dataset_type.value,
                                           config_name,
                                           annotations_ts_name,
                                           annotations_time_name,
                                           annotations_both_name,
                                           version=version.current_version)

        # If the config was not imported, save it
        if self.dataset_config.import_identifier is None or self.dataset_config.export_update_needed:
            self.save_config(export_benchmark.config_identifier, force_write=force_write, hard_force=kwargs["hard_force"])
        else:
            self.logger.info("Using already existing config with identifier: %s", self.dataset_config.import_identifier)

        # Save ts_id annotations if available and not previously imported
        if self.imported_annotations_ts_identifier is None and len(self.annotations.time_series_annotations) > 0:
            self.save_annotations(export_benchmark.annotations_ts_identifier, AnnotationType.TS_ID, force_write=force_write)
        elif self.imported_annotations_ts_identifier is not None:
            self.logger.info("Using already existing annotations with identifier: %s; type: %s", self.imported_annotations_ts_identifier, AnnotationType.TS_ID)

        # Save id_time annotations if available and not previously imported
        if self.imported_annotations_time_identifier is None and len(self.annotations.time_annotations) > 0:
            self.save_annotations(export_benchmark.annotations_time_identifier, AnnotationType.ID_TIME, force_write=force_write)
        elif self.imported_annotations_time_identifier is not None:
            self.logger.info("Using already existing annotations with identifier: %s; type: %s", self.imported_annotations_time_identifier, AnnotationType.ID_TIME)

        # Save both annotations if available and not previously imported
        if self.imported_annotations_both_identifier is None and len(self.annotations.time_in_series_annotations) > 0:
            self.save_annotations(export_benchmark.annotations_both_identifier, AnnotationType.BOTH, force_write=force_write)
        elif self.imported_annotations_both_identifier is not None:
            self.logger.info("Using already existing annotations with identifier: %s; type: %s", self.imported_annotations_both_identifier, AnnotationType.BOTH)

        # Ensure the benchmark directory exists
        if not os.path.exists(self.metadata.benchmarks_root):
            os.makedirs(self.metadata.benchmarks_root)
            self.logger.info("Created benchmarks directory at %s", self.metadata.benchmarks_root)

        benchmark_path = os.path.join(self.metadata.benchmarks_root, f"{identifier}.yaml")

        if os.path.exists(benchmark_path) and not force_write:
            self.logger.error("Benchmark file already exists at %s", benchmark_path)
            raise ValueError(f"Benchmark at path {benchmark_path} already exists. Set force_write=True to overwrite.")
        self.logger.debug("Benchmark YAML file path: %s", benchmark_path)

        yaml_dump(export_benchmark.to_dict(), benchmark_path)
        self.logger.info("Benchmark successfully saved to %s", benchmark_path)

    def get_transformers(self) -> np.ndarray[Transformer] | Transformer | None:
        """Return used transformers from config. """
        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting get transformers.")

        return self.dataset_config.transformers

    def check_errors(self) -> None:
        """
        Validates whether the dataset is corrupted. 

        Raises an exception if corrupted.
        """

        dataset, _ = load_database(self.metadata.dataset_path)

        try:
            node_iter = dataset.walk_nodes()

            # Process each node in the dataset
            for node in node_iter:
                if isinstance(node, tb.Table):

                    iter_by = min(LOADING_WARNING_THRESHOLD, len(node))
                    iters_done = 0

                    # Process the node in chunks to avoid memory issues
                    while iters_done < len(node):
                        iter_by = min(LOADING_WARNING_THRESHOLD, len(node) - iters_done)
                        _ = node[iters_done: iters_done + iter_by]  # Fetch the data in chunks
                        iters_done += iter_by

                    self.logger.info("Table '%s' checked successfully. (%d rows processed)", node._v_pathname, len(node))

            self.logger.info("Dataset check completed with no errors found.")

        except Exception as e:
            self.logger.error("Error encountered during dataset check: %s", str(e))

        finally:
            dataset.close()
            self.logger.debug("Dataset connection closed.")

    @abstractmethod
    def _get_data_for_plot(self, ts_id: int, feature_indices: np.ndarray[int], time_format: TimeFormat) -> tuple[np.ndarray, np.ndarray]:
        """Dataset type specific retrieval of data. """
        ...

    def __get_data_for_plot(self, ts_id: int, features: list[str] | str, time_format: TimeFormat) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Returns prepared data for plotting. """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before getting data for plotting.")

        features_indices = []

        if features == "config":
            features = deepcopy(self.dataset_config.features_to_take_without_ids)
            features_indices = np.arange(len(features))
            self.logger.debug("Features set from dataset config: %s", features)
        else:
            if isinstance(features, str):
                features = [features]

            if len(features) == 0:
                raise ValueError("No features specified to plot. Please provide valid features.")
            if len(set(features)) != len(features):
                raise ValueError("Duplicate features detected. All features must be unique.")

            for feature in features:
                if feature not in self.dataset_config.features_to_take_without_ids:
                    raise ValueError(f"Feature '{feature}' is not valid. It is not present in the dataset configuration.", self.dataset_config.features_to_take_without_ids)

                index_in_config_features = self.dataset_config.features_to_take_without_ids.index(feature)
                features_indices.append(index_in_config_features)

        real_feature_indices = np.array(self.dataset_config.indices_of_features_to_take_no_ids)[features_indices]
        real_feature_indices = real_feature_indices.astype(int)

        time_series, time_period = self._get_data_for_plot(ts_id, real_feature_indices, time_format)
        self.logger.debug("Time series data and corresponding time values retrieved.")

        return time_series, time_period, features

    def _validate_annotation_ids(self, ts_id: int | None, id_time: int | None) -> None:
        """Validates whether the `ts_id` and `id_time` belong to this dataset. """

        assert ts_id is not None or id_time is not None, "Either ts_id or id_time must be provided."

        # Handle when id_time is provided
        if id_time is not None:
            time_indices = self.metadata.time_indices
            if id_time < time_indices[ID_TIME_COLUMN_NAME][0] or id_time > time_indices[ID_TIME_COLUMN_NAME][-1]:
                valid_range = range(time_indices[ID_TIME_COLUMN_NAME][0], time_indices[ID_TIME_COLUMN_NAME][-1])
                raise ValueError(f"id_time {id_time} does not fall within the valid range for {self.metadata.aggregation}. "
                                 f"Valid id_time range: {valid_range}.")

        # Handle when ts_id is provided
        if ts_id is not None:
            ts_indices = self.metadata.ts_indices[self.metadata.ts_id_name]

            if ts_id not in ts_indices:
                valid_ts_range = self.metadata.ts_indices[self.metadata.ts_id_name]
                raise ValueError(f"ts_id {ts_id} does not exist in the available range for {self.metadata.source_type}. "
                                 f"Valid ts_id values: {valid_ts_range}.")

    def _get_time_based_dataloader(self, dataset: SplittedDataset, workers: int, take_all: bool, batch_size: int) -> DataLoader:
        """
        Returns a time-based PyTorch DataLoader. 

        The batch size determines the number of times for time series are included in each batch.    
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before getting dataloader.")

        if take_all:
            batch_size = len(dataset)
            self.logger.debug("Using full dataset as batch size (%d samples).", batch_size)
        else:
            self.logger.debug("Using batch size from config: %d", batch_size)

            total_batch_size = batch_size * len(dataset.ts_row_ranges)
            if total_batch_size >= LOADING_WARNING_THRESHOLD:
                self.logger.warning("The total number of samples in one batch is %d (%d time series × %d times(batch size) ). Consider lowering the batch size.", total_batch_size, len(dataset.ts_row_ranges), batch_size)

        should_drop = not take_all and self.dataset_config.sliding_window_size is not None

        batch_sampler = BatchSampler(sampler=SequentialSampler(dataset), batch_size=batch_size, drop_last=should_drop)

        dataloader = DataLoader(dataset, num_workers=0, collate_fn=self._collate_fn, persistent_workers=False, batch_size=None, sampler=batch_sampler)

        self.logger.debug("Dataloader created with SequentialSampler and batch size %d.", batch_size)

        # Prepare the dataset for loading, either with the full batch or with windowed batching
        if take_all:
            dataset.prepare_dataset(batch_size, None, None, None, workers)
            self.logger.debug("Dataset prepared with full batch size (%d samples).", batch_size)
        else:
            dataset.prepare_dataset(batch_size, self.dataset_config.sliding_window_size, self.dataset_config.sliding_window_prediction_size, self.dataset_config.sliding_window_step, workers)
            self.logger.debug("Dataset prepared with window size (%d).", self.dataset_config.sliding_window_size)

        return dataloader

    def _get_series_based_dataloader(self, dataset: SeriesBasedDataset, workers: int, take_all: bool, batch_size: int, order: DataloaderOrder = DataloaderOrder.SEQUENTIAL) -> DataLoader:
        """ 
        Returns a series-based PyTorch DataLoader.

        The batch size determines the number of time series included in each batch.
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before getting dataloader.")

        if take_all:
            batch_size = len(dataset)
            self.logger.debug("Using full dataset as batch size (%d samples) to return the entire dataset.", batch_size)
        else:
            self.logger.debug("Using batch size from config: %d", batch_size)

        total_batch_size = batch_size * len(dataset.time_period)
        if total_batch_size >= LOADING_WARNING_THRESHOLD:
            self.logger.warning("The total number of samples in one batch is %d (%d time series(batch size) × %d times ). Consider lowering the batch size.", total_batch_size, batch_size, len(dataset.time_period))

        if order == DataloaderOrder.RANDOM:
            if self.dataset_config.random_state is not None:
                generator = torch.Generator()
                generator.manual_seed(self.dataset_config.random_state)
                self.logger.debug("Prepared RandomSampler with fixed seed %d for series dataloader.", self.dataset_config.random_state)
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
        dataloader = DataLoader(dataset, num_workers=workers, collate_fn=self._collate_fn, worker_init_fn=SeriesBasedDataset.worker_init_fn, persistent_workers=False, batch_size=None, sampler=batch_sampler)

        # Must be done if dataloader runs on main process.
        if workers == 0:
            dataset.pytables_worker_init(0)

        self.logger.debug("Series-based dataset prepared for dataloader with batch size %d and %s order.", batch_size, order.name)

        return dataloader

    @abstractmethod
    def _get_singular_time_series_dataset(self, parent_dataset: SeriesBasedDataset | SplittedDataset, ts_id: int) -> SeriesBasedDataset | SplittedDataset:
        """Returns dataset for single time series """
        ...

    @abstractmethod
    def _get_dataloader(self, dataset: Dataset, workers: int, take_all: bool, batch_size: int, **kwargs) -> DataLoader:
        """
        Sets the DataLoader based on the type of dataset. 

        This method determines whether the dataset is series-based or time-based and calls the corresponding method to return the appropriate DataLoader:
        - Calls [`_get_series_based_dataloader`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset._get_series_based_dataloader] if the dataset is series-based.
        - Calls [`_get_time_based_dataloader`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset._get_time_based_dataloader] if the dataset is time-based.
        """
        ...

    def _get_df(self, dataloader: DataLoader, as_single_dataframe: bool, ts_ids: np.ndarray, time_period: np.ndarray) -> pd.DataFrame:
        """Returns all data from the DataLoader as a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before getting DataFrame.")

        total_samples = len(ts_ids) * len(time_period)
        if total_samples >= LOADING_WARNING_THRESHOLD:
            self.logger.warning("The dataset contains %d samples (%d time series × %d times). Consider using get_*_dataloader() for batch loading.", total_samples, len(ts_ids), len(time_period))

        if as_single_dataframe:
            self.logger.debug("Returning a single DataFrame with all features for all time series.")
            return dataset_loaders.create_single_df_from_dataloader(
                dataloader,
                ts_ids,
                self.dataset_config.features_to_take,
                self.dataset_config.time_format,
                self.dataset_config.include_ts_id,
                self.dataset_config.include_time,
                self.dataset_config.dataset_type,
                True
            )
        else:
            self.logger.debug("Returning multiple DataFrames, one per time series.")
            return dataset_loaders.create_multiple_df_from_dataloader(
                dataloader,
                ts_ids,
                self.dataset_config.features_to_take,
                self.dataset_config.time_format,
                self.dataset_config.include_ts_id,
                self.dataset_config.include_time,
                self.dataset_config.dataset_type,
                True
            )

    def _get_numpy(self, dataloader: DataLoader, ts_ids: np.ndarray, time_period: np.ndarray) -> np.ndarray:
        """Returns all data from the DataLoader as a NumPy `ndarray`. """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before getting Numpy array.")

        total_samples = len(ts_ids) * len(time_period)
        if total_samples >= LOADING_WARNING_THRESHOLD:
            self.logger.warning("The dataset contains %d samples (%d time series × %d times). Consider using get_*_dataloader() for batch loading.", total_samples, len(ts_ids), len(time_period))

        self.logger.debug("Creating numpy array from dataloader.")
        return dataset_loaders.create_numpy_from_dataloader(
            dataloader,
            ts_ids,
            self.dataset_config.time_format,
            self.dataset_config.include_time,
            self.dataset_config.dataset_type,
            True
        )

    def _clear(self) -> None:
        """Clears set data. Mainly called when initializing new config. """
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        self.all_dataset = None
        self.all_dataloader = None
        self.dataset_config = None
        self.logger.debug("Dataset attributes had been cleared. ")

    def _update_annotations_imported_status(self, on: AnnotationType, identifier: str):
        if on == AnnotationType.TS_ID:
            self.imported_annotations_ts_identifier = identifier
        elif on == AnnotationType.ID_TIME:
            self.imported_annotations_time_identifier = identifier
        elif on == AnnotationType.BOTH:
            self.imported_annotations_both_identifier = identifier

    def _update_config_imported_status(self, identifier: str) -> None:
        self.dataset_config.import_identifier = identifier
        self._export_config_copy.import_identifier = identifier

    @abstractmethod
    def _initialize_datasets(self) -> None:
        """ Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize], initializes datasets for sets."""
        ...

    @abstractmethod
    def _initialize_transformers_and_details(self, workers: int) -> None:
        """ Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize]. Goes through data to validate time series against `nan_threshold`, fit `transformers`, fit `anomaly handlers` and prepare `fillers`"""
        ...

    def _update_export_config_copy(self) -> None:
        """ Called at the end of [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize] or when changing config values. Updates values of config used for saving config."""

        self._export_config_copy.train_batch_size = self.dataset_config.train_batch_size
        self._export_config_copy.val_batch_size = self.dataset_config.val_batch_size
        self._export_config_copy.test_batch_size = self.dataset_config.test_batch_size
        self._export_config_copy.all_batch_size = self.dataset_config.all_batch_size

        self._export_config_copy.train_workers = self.dataset_config.train_workers
        self._export_config_copy.val_workers = self.dataset_config.val_workers
        self._export_config_copy.test_workers = self.dataset_config.test_workers
        self._export_config_copy.all_workers = self.dataset_config.all_workers
        self._export_config_copy.init_workers = self.dataset_config.init_workers

    def _validate_config_for_dataset(self, config: DatasetConfig) -> bool:
        """Validates whether config is supposed to be used for this dataset. """

        if config.database_name != self.metadata.database_name:
            self.logger.error("This config is not compatible with the current dataset. Difference in database name between config and this dataset.")
            raise ValueError("This config is not compatible with the current dataset.", f"config.database_name == {config.database_name} and dataset.database_name == {self.metadata.database_name}")

        if config.dataset_type != self.metadata.dataset_type:
            self.logger.error("This config is not compatible with the current dataset. Difference in is_series_based between config and this dataset.")
            raise ValueError("This config is not compatible with the current dataset.", f"config.dataset_type == {config.dataset_type} and dataset.dataset_type == {self.metadata.dataset_type}")

        if config.aggregation != self.metadata.aggregation:
            self.logger.error("This config is not compatible with the current dataset. Difference in aggregation type between config and this dataset.")
            raise ValueError("This config is not compatible with the current dataset.", f"config.aggregation == {config.aggregation} and dataset.aggregation == {self.metadata.aggregation}")

        if config.source_type != self.metadata.source_type:
            self.logger.error("This config is not compatible with the current dataset. Difference in source type between config and this dataset.")
            raise ValueError("This config is not compatible with the current dataset.", f"config.source_type == {config.source_type} and dataset.source_type == {self.metadata.source_type}")
