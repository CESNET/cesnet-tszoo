from datetime import datetime, timezone
from typing import Optional, Literal
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from cesnet_tszoo.utils.enums import SplitType, TimeFormat
from cesnet_tszoo.configs.time_based_config import TimeBasedConfig
from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.pytables_data.time_based_initializer_dataset import TimeBasedInitializerDataset
from cesnet_tszoo.pytables_data.splitted_dataset import SplittedDataset
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME


@dataclass
class TimeBasedCesnetDataset(CesnetDataset):
    """This class is used for time-based returning of data. Can be created by using [`get_dataset`][cesnet_tszoo.datasets.cesnet_database.CesnetDatabase.get_dataset] with parameter `is_series_based` = `False`.

    Time-based means batch size affects number of returned times in one batch. Which time series are returned does not change.

    The dataset provides multiple ways to access the data:

    - **Iterable PyTorch DataLoader**: For batch processing.
    - **Pandas DataFrame**: For loading the entire training, validation, test or all set at once.
    - **Numpy array**: For loading the entire training, validation, test or all set at once. 
    - See [loading data][loading-data] for more details.

    The dataset is stored in a [PyTables](https://www.pytables.org/) database. The internal `TimeBasedDataset`, `SplittedDataset`, `TimeBasedInitializerDataset` classes (used only when calling [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]) act as wrappers that implement the PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 
    interface. These wrappers are compatible with PyTorch’s [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), providing efficient parallel data loading. 

    The dataset configuration is done through the [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig] class.       

    **Intended usage:**

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.cesnet_database.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test/test_other), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy]. 
    6. (Optional) Evaluate the model on [`get_test_other_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_dataloader]/[`get_test_other_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_df]/[`get_test_other_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_numpy].    

    Alternatively you can use [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark]

    1. Call [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark] with the desired benchmark. You can use your own saved benchmark or you can use already built-in one. This will download the dataset and annotations (if available) if they have not been previously downloaded.
    2. Retrieve the initialized dataset using [`get_initialized_dataset`][cesnet_tszoo.benchmarks.Benchmark.get_initialized_dataset]. This will provide a dataset that is ready to use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy]. 
    6. (Optional) Evaluate the model on [`get_test_other_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_dataloader]/[`get_test_other_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_df]/[`get_test_other_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_numpy].    

    Parameters:
        database_name: Name of the database.
        dataset_path: Path to the dataset file.     
        configs_root: Path to the folder where configurations are saved.
        benchmarks_root: Path to the folder where benchmarks are saved.
        annotations_root: Path to the folder where annotations are saved.
        source_type: The source type of the dataset.
        aggregation: The aggregation type for the selected source type.
        ts_id_name: Name of the id used for time series.
        default_values: Default values for each available feature.
        additional_data: Available small datasets. Can get them by calling [`get_additional_data`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_additional_data] with their name.

    Attributes:
        time_indices: Available time IDs for the dataset.
        ts_indices: Available time series IDs for the dataset.
        annotations: Annotations for the selected dataset.
        logger: Logger for displaying information.  
        imported_annotations_ts_identifier: Identifier for the imported annotations of type `AnnotationType.TS_ID`.
        imported_annotations_time_identifier: Identifier for the imported annotations of type `AnnotationType.ID_TIME`.
        imported_annotations_both_identifier: Identifier for the imported annotations of type `AnnotationType.BOTH`.  

    The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize] is called.

    Attributes:
        dataset_config: Configuration of the dataset.
        train_dataset: Training set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.
        val_dataset: Validation set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.
        test_dataset: Test set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.  
        test_other_dataset: Test_other set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.
        all_dataset: All set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.
        train_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training set.
        val_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation set.
        test_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test set.
        test_other_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test_other set.            
        all_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for all set.        
    """

    dataset_config: Optional[TimeBasedConfig] = field(default=None, init=False)

    train_dataset: Optional[SplittedDataset] = field(default=None, init=False)
    val_dataset: Optional[SplittedDataset] = field(default=None, init=False)
    test_dataset: Optional[SplittedDataset] = field(default=None, init=False)
    test_other_dataset: Optional[SplittedDataset] = field(default=None, init=False)
    all_dataset: Optional[SplittedDataset] = field(default=None, init=False)

    train_dataloader: Optional[DataLoader] = field(default=None, init=False)
    val_dataloader: Optional[DataLoader] = field(default=None, init=False)
    test_dataloader: Optional[DataLoader] = field(default=None, init=False)
    test_other_dataloader: Optional[DataLoader] = field(default=None, init=False)
    all_dataloader: Optional[DataLoader] = field(default=None, init=False)

    is_series_based: bool = field(default=False, init=False)

    _export_config_copy: Optional[TimeBasedConfig] = field(default=None, init=False)

    def set_dataset_config_and_initialize(self, dataset_config: TimeBasedConfig, display_config_details: bool = True, workers: int | Literal["config"] = "config") -> None:
        """
        Initialize training set, validation est, test set etc.. This method must be called before any data can be accessed. It is required for the final initialization of [`dataset_config`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig].

        The following configuration attributes are used during initialization:

        | Dataset config                    | Description                                                                                    |
        | --------------------------------- | ---------------------------------------------------------------------------------------------- |
        | `init_workers`                    | Specifies the number of workers to use for initialization. Applied when `workers` = "config".  |
        | `partial_fit_initialized_transformers` | Determines whether initialized transformers should be partially fitted on the training data.        |
        | `nan_threshold`                   | Filters out time series with missing values exceeding the specified threshold.                 |

        Parameters:
            dataset_config: Desired configuration of the dataset.
            display_config_details: Flag indicating whether to display the configuration values after initialization. `Default: True`  
            workers: The number of workers to use during initialization. `Default: "config"`  
        """

        assert isinstance(dataset_config, TimeBasedConfig), "TimeBasedCesnetDataset can only use TimeBasedConfig."

        super(TimeBasedCesnetDataset, self).set_dataset_config_and_initialize(dataset_config, display_config_details, workers)

    def get_data_about_set(self, about: SplitType | Literal["train", "val", "test", "all"]) -> dict:
        """
        Retrieve data related to the specified set.

        Parameters:
            about: Specifies the set to retrieve data about.

        Returned dictionary contains:

        - **ts_ids:** Ids of time series in `about` set.
        - **test_ts_ids:** Ids of  time series in `test_ts_ids`. Only for `about` == SplitType.TEST and when `test_ts_id` is set in used config.
        - **TimeFormat.ID_TIME:** Times in `about` set, where time format is `TimeFormat.ID_TIME`.
        - **TimeFormat.DATETIME:** Times in `about` set, where time format is `TimeFormat.DATETIME`.
        - **TimeFormat.UNIX_TIME:** Times in `about` set, where time format is `TimeFormat.UNIX_TIME`.
        - **TimeFormat.SHIFTED_UNIX_TIME:** Times in `about` set, where time format is `TimeFormat.SHIFTED_UNIX_TIME`.

        Returns:
            Returns dictionary with details about set.
        """
        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting data about set.")

        about = SplitType(about)

        time_period = None

        result = {}

        if about == SplitType.TRAIN:
            if not self.dataset_config.has_train:
                raise ValueError("Train split is not used.")
            time_period = self.dataset_config.train_time_period
        elif about == SplitType.VAL:
            if not self.dataset_config.has_val:
                raise ValueError("Val split is not used.")
            time_period = self.dataset_config.val_time_period
        elif about == SplitType.TEST:
            if not self.dataset_config.has_test:
                raise ValueError("Test split is not used.")
            time_period = self.dataset_config.test_time_period
            result["test_ts_ids"] = self.dataset_config.test_ts_ids.copy() if self.dataset_config.test_ts_ids is not None else None
        elif about == SplitType.ALL:
            if not self.dataset_config.has_all:
                raise ValueError("All split is not used.")

            time_period = self.dataset_config.all_time_period
        else:
            raise ValueError("Invalid split type!")

        datetime_temp = np.array([datetime.fromtimestamp(time, timezone.utc) for time in self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]]])

        result["ts_ids"] = self.dataset_config.ts_ids.copy()
        result[TimeFormat.ID_TIME] = time_period[ID_TIME_COLUMN_NAME].copy()
        result[TimeFormat.DATETIME] = datetime_temp.copy()
        result[TimeFormat.UNIX_TIME] = self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]].copy()
        result[TimeFormat.SHIFTED_UNIX_TIME] = self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]] - self.time_indices[TIME_COLUMN_NAME][0]

        return result

    def get_test_other_dataloader(self, ts_id: int | None = None, workers: int | Literal["config"] = "config", **kwargs) -> DataLoader:
        """
        Returns a PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test_other set.

        The `DataLoader` is created on the first call and cached for subsequent use. <br/>
        The cached dataloader is cleared when either [`get_test_other_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_df] or [`get_test_other_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_numpy] is called.

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
        | `test_batch_size`                  | Number of times for time series per batch.                                                                                                |
        | `sliding_window_size`              | Modifies the shape of the returned data.                                                                                                  |
        | `sliding_window_prediction_size`   | Modifies the shape of the returned data.                                                                                                  |
        | `sliding_window_step`              | Available only for time-based datasets. Number of times to move by after each window.                                                     |
        | `test_workers`                     | Specifies the number of workers to use for loading test_other data. Applied when `workers` = "config".                                    |

        Parameters:
            workers: The number of workers to use for loading test_other data. `Default: "config"`  
            ts_id: Specifies time series to take. If None returns all time series as normal. `Default: "None"`

        Returns:
            An iterable `DataLoader` containing data from test_other set.         
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access test_other_dataloader.")

        if not self.dataset_config.has_test or not self.dataset_config.has_test_ts_ids:
            raise ValueError("Dataloader for test_other set is not available in the dataset configuration.")

        assert self.test_dataset is not None, "The test_other_dataset must be initialized before accessing data from test_other set."

        default_kwargs = {'take_all': False, "cache_loader": True}
        kwargs = {**default_kwargs, **kwargs}

        if ts_id is not None:

            if ts_id == self.dataset_config.used_singular_test_other_time_series and self.test_other_dataloader is not None:
                self.logger.debug("Returning cached test_other_dataloader.")
                return self.test_other_dataloader

            dataset = self._get_singular_time_series_dataset(self.test_other_dataset, ts_id)
            self.dataset_config.used_singular_test_other_time_series = ts_id
            if self.test_other_dataloader:
                del self.test_other_dataloader
                self.test_other_dataloader = None
                self.logger.info("Destroyed previous cached test_other_dataloader.")

            self.dataset_config.used_test_other_workers = 0
            self.test_other_dataloader = self._get_dataloader(dataset, 0, False, self.dataset_config.test_batch_size)
            self.logger.info("Created new cached test_other_dataloader.")
            return self.test_other_dataloader
        elif self.dataset_config.used_singular_test_other_time_series is not None and self.test_other_dataloader is not None:
            del self.test_other_dataloader
            self.test_other_dataloader = None
            self.dataset_config.used_singular_test_other_time_series = None
            self.logger.info("Destroyed previous cached test_other_dataloader.")

        if workers == "config":
            workers = self.dataset_config.test_workers

        # If the dataloader is cached and number of used workers did not change, return the cached dataloader
        if self.test_other_dataloader and kwargs["cache_loader"] and workers == self.dataset_config.used_test_other_workers:
            self.logger.debug("Returning cached test_other_dataloader.")
            return self.test_other_dataloader

        # Update the used workers count
        self.dataset_config.used_test_other_workers = workers

        # If there's a previously cached dataloader, destroy it
        if self.test_other_dataloader:
            del self.test_other_dataloader
            self.test_other_dataloader = None
            self.logger.info("Destroyed previous cached test_other_dataloader.")

        # If caching is enabled, create a new cached dataloader
        if kwargs["cache_loader"]:
            self.test_other_dataloader = self._get_dataloader(self.test_other_dataset, workers, kwargs['take_all'], self.dataset_config.test_batch_size)
            self.logger.info("Created new cached test_other_dataloader.")
            return self.test_other_dataloader

        # If caching is disabled, create a new uncached dataloader
        self.logger.debug("Created new uncached test_other_dataloader.")
        return self._get_dataloader(self.test_other_dataset, workers, kwargs['take_all'], self.dataset_config.test_batch_size)

    def get_test_other_df(self, workers: int | Literal["config"] = "config", as_single_dataframe: bool = True) -> pd.DataFrame:
        """
        Creates a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing all the data from test_other set grouped by time series.

        This method uses the `test_other_dataloader` with a batch size set to the total number of data in the test_other set. The cached `test_other_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire test_other set is loaded into memory, which may lead to high memory usage. If working with large test_other set, consider using `get_test_other_dataloader` instead to handle data in batches.

        Parameters:
            workers: The number of workers to use for loading test_other data. `Default: "config"`  
            as_single_dataframe: Whether to return a single dataframe with all time series combined, or to create separate dataframes for each time series. `Default: True` 

        Returns:
            A single Pandas DataFrame containing all data from test_other set, or a list of DataFrames (one per time series).
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access test_other_dataloader.")

        if not self.dataset_config.has_test or not self.dataset_config.has_test_ts_ids:
            raise ValueError("Dataloader for test_other set is not available in the dataset configuration.")

        assert self.test_dataset is not None, "The test_other_dataset must be initialized before accessing data from test_other set."

        ts_ids, time_period = self.dataset_config._get_test_other()

        dataloader = self.get_test_other_dataloader(workers=workers, take_all=True, cache_loader=False)
        return self._get_df(dataloader, as_single_dataframe, ts_ids, time_period)

    def get_test_other_numpy(self, workers: int | Literal["config"] = "config",) -> np.ndarray:
        """
        Creates a NumPy array containing all the data from test_other set grouped by time series, with the shape `(num_time_series, num_times, num_features)`.

        This method uses the `test_other_dataloader` with a batch size set to the total number of data in the test_other set. The cached `test_other_dataloader` is cleared during this operation.

        !!! warning "Memory usage"
            The entire test_other set is loaded into memory, which may lead to high memory usage. If working with large test_other set, consider using `get_test_other_dataloader` instead to handle data in batches.        

        Parameters:
            workers: The number of workers to use for loading test_other data. `Default: "config"`  

        Returns:
            A NumPy array containing all the data in test_other set with the shape `(num_time_series, num_times, num_features)`.
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized. Please call set_dataset_config_and_initialize() before attempting to access test_other_dataloader.")

        if not self.dataset_config.has_all:
            raise ValueError("Dataloader for test_other set is not available in the dataset configuration.")

        assert self.test_dataset is not None, "The test_other_dataset must be initialized before accessing data from test_other set."

        ts_ids, time_period = self.dataset_config._get_test_other()

        dataloader = self.get_test_other_dataloader(workers=workers, take_all=True, cache_loader=False)
        return self._get_numpy(dataloader, ts_ids, time_period)

    def set_sliding_window(self, sliding_window_size: int | None | Literal["config"] = "config", sliding_window_prediction_size: int | None | Literal["config"] = "config",
                           sliding_window_step: int | None | Literal["config"] = "config", set_shared_size: float | int | Literal["config"] = "config", workers: int | Literal["config"] = "config") -> None:
        """Used for updating sliding window related values set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                     | Description                                                                                                                                     |
        | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
        | `sliding_window_size`              | Number of times in one window. Impacts dataloader behavior. Refer to relevant config for details.                                               |
        | `sliding_window_prediction_size`   | Number of times to predict from sliding_window_size. Refer to relevant config for details.                                                      |
        | `sliding_window_step`              | Number of times to move by after each window. Refer to relevant config for details.                                                             |
        | `set_shared_size`                  | How much times should time periods share. Order of sharing is training set < validation set < test set. Refer to relevant config for details.   |        

        Parameters:
            sliding_window_size: Number of times in one window. `Defaults: config`.
            sliding_window_prediction_size: Number of times to predict from sliding_window_size. `Defaults: config`.
            sliding_window_step: Number of times to move by after each window. `Defaults: config`.
            set_shared_size: How much times should time periods share. `Defaults: config`.
            workers: How many workers to use when setting new sliding window values. `Defaults: config`.  
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating sliding window values.")

        self.update_dataset_config_and_initialize(sliding_window_size=sliding_window_size, sliding_window_prediction_size=sliding_window_prediction_size, sliding_window_step=sliding_window_step, set_shared_size=set_shared_size, workers=workers)
        self.logger.info("Sliding window values has been changed successfuly.")

    def _initialize_datasets(self) -> None:
        """Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize], this method initializes the set datasets (train, validation, test, test_other and all). """

        if self.dataset_config.has_train:
            self.train_dataset = SplittedDataset(self.dataset_path,
                                                 self.dataset_config._get_table_data_path(),
                                                 self.dataset_config.ts_id_name,
                                                 self.dataset_config.ts_row_ranges,
                                                 self.dataset_config.train_time_period,
                                                 self.dataset_config.features_to_take,
                                                 self.dataset_config.indices_of_features_to_take_no_ids,
                                                 self.dataset_config.default_values,
                                                 self.dataset_config.train_fillers,
                                                 self.dataset_config.create_transformer_per_time_series,
                                                 self.dataset_config.include_time,
                                                 self.dataset_config.include_ts_id,
                                                 self.dataset_config.time_format,
                                                 self.dataset_config.train_workers,
                                                 self.dataset_config.transformers)
            self.logger.debug("train_dataset initiliazed.")

        if self.dataset_config.has_val:
            self.val_dataset = SplittedDataset(self.dataset_path,
                                               self.dataset_config._get_table_data_path(),
                                               self.dataset_config.ts_id_name,
                                               self.dataset_config.ts_row_ranges,
                                               self.dataset_config.val_time_period,
                                               self.dataset_config.features_to_take,
                                               self.dataset_config.indices_of_features_to_take_no_ids,
                                               self.dataset_config.default_values,
                                               self.dataset_config.val_fillers,
                                               self.dataset_config.create_transformer_per_time_series,
                                               self.dataset_config.include_time,
                                               self.dataset_config.include_ts_id,
                                               self.dataset_config.time_format,
                                               self.dataset_config.val_workers,
                                               self.dataset_config.transformers)
            self.logger.debug("val_dataset initiliazed.")

        if self.dataset_config.has_test:
            self.test_dataset = SplittedDataset(self.dataset_path,
                                                self.dataset_config._get_table_data_path(),
                                                self.dataset_config.ts_id_name,
                                                self.dataset_config.ts_row_ranges,
                                                self.dataset_config.test_time_period,
                                                self.dataset_config.features_to_take,
                                                self.dataset_config.indices_of_features_to_take_no_ids,
                                                self.dataset_config.default_values,
                                                self.dataset_config.test_fillers,
                                                self.dataset_config.create_transformer_per_time_series,
                                                self.dataset_config.include_time,
                                                self.dataset_config.include_ts_id,
                                                self.dataset_config.time_format,
                                                self.dataset_config.test_workers,
                                                self.dataset_config.transformers)
            self.logger.debug("test_dataset initiliazed.")

        if self.dataset_config.has_all:
            self.all_dataset = SplittedDataset(self.dataset_path,
                                               self.dataset_config._get_table_data_path(),
                                               self.dataset_config.ts_id_name,
                                               self.dataset_config.ts_row_ranges,
                                               self.dataset_config.all_time_period,
                                               self.dataset_config.features_to_take,
                                               self.dataset_config.indices_of_features_to_take_no_ids,
                                               self.dataset_config.default_values,
                                               self.dataset_config.all_fillers,
                                               self.dataset_config.create_transformer_per_time_series,
                                               self.dataset_config.include_time,
                                               self.dataset_config.include_ts_id,
                                               self.dataset_config.time_format,
                                               self.dataset_config.all_workers,
                                               self.dataset_config.transformers)
            self.logger.debug("all_dataset initiliazed.")

        if self.dataset_config.has_test_ts_ids:
            test_other_transformer = None if self.dataset_config.create_transformer_per_time_series else self.dataset_config.transformers
            self.test_other_dataset = SplittedDataset(self.dataset_path,
                                                      self.dataset_config._get_table_data_path(),
                                                      self.dataset_config.ts_id_name,
                                                      self.dataset_config.test_ts_row_ranges,
                                                      self.dataset_config.test_time_period,
                                                      self.dataset_config.features_to_take,
                                                      self.dataset_config.indices_of_features_to_take_no_ids,
                                                      self.dataset_config.default_values,
                                                      self.dataset_config.other_test_fillers,
                                                      self.dataset_config.create_transformer_per_time_series,
                                                      self.dataset_config.include_time,
                                                      self.dataset_config.include_ts_id,
                                                      self.dataset_config.time_format,
                                                      self.dataset_config.test_workers,
                                                      test_other_transformer)
            self.logger.debug("test_other_dataset initiliazed.")

    def _initialize_transformers_and_details(self, workers: int) -> None:
        """
        Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]. 

        Goes through data to validate time series against `nan_threshold`, fit/partial fit `transformers` and prepare `fillers`.
        """

        init_dataset = TimeBasedInitializerDataset(self.dataset_path,
                                                   self.dataset_config._get_table_data_path(),
                                                   self.dataset_config.ts_id_name,
                                                   self.dataset_config.ts_row_ranges,
                                                   self.dataset_config.all_time_period,
                                                   self.dataset_config.train_time_period,
                                                   self.dataset_config.val_time_period,
                                                   self.dataset_config.test_time_period,
                                                   self.dataset_config.features_to_take,
                                                   self.dataset_config.indices_of_features_to_take_no_ids,
                                                   self.dataset_config.default_values,
                                                   self.dataset_config.train_fillers,
                                                   self.dataset_config.val_fillers,
                                                   self.dataset_config.test_fillers)

        sampler = SequentialSampler(init_dataset)
        dataloader = DataLoader(init_dataset, num_workers=workers, collate_fn=self._collate_fn, worker_init_fn=TimeBasedInitializerDataset.worker_init_fn, persistent_workers=False, sampler=sampler)

        if workers == 0:
            init_dataset.pytables_worker_init()

        ts_ids_to_take = []

        self.logger.info("Updating config on train/val/test/all and selected time series.")
        for i, data in enumerate(tqdm(dataloader, total=len(self.dataset_config.ts_row_ranges))):
            train_data, train_count_values, val_count_values, test_count_values, all_count_values, val_filler, test_filler = data[0]

            missing_train_percentage = 0
            missing_val_percentage = 0
            missing_test_percentage = 0
            missing_all_percentage = 0

            # Filter time series based on missing data threshold
            if self.dataset_config.has_train:
                missing_train_percentage = train_count_values[1] / (train_count_values[0] + train_count_values[1])
            if self.dataset_config.has_val:
                missing_val_percentage = val_count_values[1] / (val_count_values[0] + val_count_values[1])
            if self.dataset_config.has_test:
                missing_test_percentage = test_count_values[1] / (test_count_values[0] + test_count_values[1])
            if self.dataset_config.has_all:
                missing_all_percentage = all_count_values[1] / (all_count_values[0] + all_count_values[1])

            if max(missing_train_percentage, missing_val_percentage, missing_test_percentage, missing_all_percentage) <= self.dataset_config.nan_threshold:
                ts_ids_to_take.append(i)

                # Fit transformers if required
                if self.dataset_config.transform_with is not None and train_data is not None and (not self.dataset_config.are_transformers_premade or self.dataset_config.partial_fit_initialized_transformers):

                    if self.dataset_config.are_transformers_premade and self.dataset_config.partial_fit_initialized_transformers:
                        if self.dataset_config.create_transformer_per_time_series:
                            self.dataset_config.transformers[i].partial_fit(train_data)
                        else:
                            self.dataset_config.transformers.partial_fit(train_data)
                    else:
                        if self.dataset_config.create_transformer_per_time_series:
                            self.dataset_config.transformers[i].fit(train_data)
                        else:
                            self.dataset_config.transformers.partial_fit(train_data)

                # Only update fillers for val/test because train doesnt need to know about previous data
                if self.dataset_config.fill_missing_with is not None:
                    if self.dataset_config.has_val:
                        self.dataset_config.val_fillers[i] = val_filler
                    if self.dataset_config.has_test:
                        self.dataset_config.test_fillers[i] = test_filler

        if workers == 0:
            init_dataset.cleanup()

        if len(ts_ids_to_take) == 0:
            raise ValueError("No valid time series left in ts_ids after applying nan_threshold.")

        # Update config based on filtered time series
        self.dataset_config.ts_row_ranges = self.dataset_config.ts_row_ranges[ts_ids_to_take]
        self.dataset_config.ts_ids = self.dataset_config.ts_ids[ts_ids_to_take]

        if self.dataset_config.transform_with is not None:
            if self.dataset_config.create_transformer_per_time_series:
                self.dataset_config.transformers = self.dataset_config.transformers[ts_ids_to_take]

        if self.dataset_config.fill_missing_with is not None:
            if self.dataset_config.has_train:
                self.dataset_config.train_fillers = self.dataset_config.train_fillers[ts_ids_to_take]
            if self.dataset_config.has_val:
                self.dataset_config.val_fillers = self.dataset_config.val_fillers[ts_ids_to_take]
            if self.dataset_config.has_test:
                self.dataset_config.test_fillers = self.dataset_config.test_fillers[ts_ids_to_take]
            if self.dataset_config.has_all:
                self.dataset_config.all_fillers = self.dataset_config.all_fillers[ts_ids_to_take]

        self.dataset_config.used_ts_row_ranges = self.dataset_config.ts_row_ranges
        self.dataset_config.used_ts_ids = self.dataset_config.ts_ids
        self.dataset_config.used_times = self.dataset_config.all_time_period
        self.dataset_config.used_fillers = self.dataset_config.all_fillers

        self.logger.debug("ts_ids updated: %s time series left.", len(ts_ids_to_take))

        # Check if going through test_other data is needed
        if self.dataset_config.test_ts_row_ranges is not None:
            test_init_dataset = TimeBasedInitializerDataset(self.dataset_path,
                                                            self.dataset_config._get_table_data_path(),
                                                            self.dataset_config.ts_id_name,
                                                            self.dataset_config.test_ts_row_ranges,
                                                            None,
                                                            None,
                                                            None,
                                                            self.dataset_config.test_time_period,
                                                            self.dataset_config.features_to_take,
                                                            self.dataset_config.indices_of_features_to_take_no_ids,
                                                            self.dataset_config.default_values,
                                                            None,
                                                            None,
                                                            self.dataset_config.other_test_fillers)

            sampler = SequentialSampler(test_init_dataset)
            dataloader = DataLoader(test_init_dataset, num_workers=workers, collate_fn=self._collate_fn, worker_init_fn=TimeBasedInitializerDataset.worker_init_fn, persistent_workers=False, sampler=sampler)

            if workers == 0:
                test_init_dataset.pytables_worker_init()

            test_ts_ids_to_take = []

            self.logger.info("Updating config on test_other and selected time series.")
            for i, data in enumerate(tqdm(dataloader, total=len(self.dataset_config.test_ts_row_ranges))):
                _, _, _, test_count_values, _, _, test_filler = data[0]

                missing_test_percentage = test_count_values[1] / (test_count_values[0] + test_count_values[1])

                if missing_test_percentage <= self.dataset_config.nan_threshold:
                    test_ts_ids_to_take.append(i)

                    if self.dataset_config.fill_missing_with is not None:
                        self.dataset_config.other_test_fillers[i] = test_filler

            if workers == 0:
                test_init_dataset.cleanup()

            if len(test_ts_ids_to_take) == 0:
                raise ValueError("No valid time series left in test_ts_ids after applying nan_threshold.")

            # Update config based on filtered time series
            self.dataset_config.test_ts_row_ranges = self.dataset_config.test_ts_row_ranges[test_ts_ids_to_take]
            self.dataset_config.test_ts_ids = self.dataset_config.test_ts_ids[test_ts_ids_to_take]

            if self.dataset_config.fill_missing_with is not None:
                if self.dataset_config.other_test_fillers is not None:
                    self.dataset_config.other_test_fillers = self.dataset_config.other_test_fillers[test_ts_ids_to_take]

            self.dataset_config.used_ts_row_ranges = np.concatenate((self.dataset_config.used_ts_row_ranges, self.dataset_config.test_ts_row_ranges))
            self.dataset_config.used_ts_ids = np.concatenate((self.dataset_config.used_ts_ids, self.dataset_config.test_ts_ids))

            if self.dataset_config.fill_missing_with is not None:
                self.dataset_config.used_fillers = np.concatenate((self.dataset_config.used_fillers, self.dataset_config.other_test_fillers))

            self.logger.debug("test_ts_ids updated: %s time series left.", len(test_ts_ids_to_take))

    def _update_export_config_copy(self) -> None:
        """
        Called at the end of [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize] or when changing config values. 

        Updates values of config used for saving config.
        """
        self._export_config_copy.database_name = self.database_name

        if self.dataset_config.has_train:
            self._export_config_copy.ts_ids = self.dataset_config.ts_ids.copy()
            self.logger.debug("Updated ts_ids of _export_config_copy.")

        if self.dataset_config.has_test_ts_ids:
            self._export_config_copy.test_ts_ids = self.dataset_config.test_ts_ids.copy()
            self.logger.debug("Updated test_ts_ids of _export_config_copy")

        super(TimeBasedCesnetDataset, self)._update_export_config_copy()

    def _get_singular_time_series_dataset(self, parent_dataset: SplittedDataset, ts_id: int) -> SplittedDataset:
        """Returns dataset for single time series """

        temp = np.where(np.isin(parent_dataset.ts_row_ranges[self.ts_id_name], [ts_id]))[0]

        if len(temp) == 0:
            raise ValueError(f"ts_id {ts_id} was not found in valid time series for this set. Available time series are: {parent_dataset.ts_row_ranges[self.ts_id_name]}")

        time_series_position = temp[0]

        filler = None if parent_dataset.fillers is None else parent_dataset.fillers[time_series_position:time_series_position + 1]

        transformer = None
        if parent_dataset.feature_transformers is not None:
            transformer = parent_dataset.feature_transformers[time_series_position:time_series_position + 1] if parent_dataset.is_transformer_per_time_series else parent_dataset.feature_transformers

        dataset = SplittedDataset(self.dataset_path,
                                  self.dataset_config._get_table_data_path(),
                                  self.dataset_config.ts_id_name,
                                  parent_dataset.ts_row_ranges[time_series_position: time_series_position + 1],
                                  parent_dataset.time_period,
                                  self.dataset_config.features_to_take,
                                  self.dataset_config.indices_of_features_to_take_no_ids,
                                  self.dataset_config.default_values,
                                  filler,
                                  self.dataset_config.create_transformer_per_time_series,
                                  self.dataset_config.include_time,
                                  self.dataset_config.include_ts_id,
                                  self.dataset_config.time_format,
                                  0,
                                  transformer)
        self.logger.debug("Singular time series dataset initiliazed.")

        return dataset

    def _get_dataloader(self, dataset: SplittedDataset, workers: int | Literal["config"], take_all: bool, batch_size: int, **kwargs) -> DataLoader:
        """ Set time based dataloader for this dataset. """

        return self._get_time_based_dataloader(dataset, workers, take_all, batch_size)

    def _clear(self) -> None:
        """ Clears set data. Mainly called when initializing new config. """

        self.test_other_dataloader = None
        self.test_other_dataset = None
        super(TimeBasedCesnetDataset, self)._clear()
