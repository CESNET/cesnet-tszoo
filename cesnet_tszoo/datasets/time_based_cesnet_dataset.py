from datetime import datetime, timezone
from typing import Optional, Literal
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from cesnet_tszoo.utils.enums import SplitType, TimeFormat, DatasetType
from cesnet_tszoo.configs.time_based_config import TimeBasedConfig
from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.pytables_data.datasets.time_based import TimeBasedSplittedDataset, TimeBasedDataloader, TimeBasedDataloaderFactory, TimeBasedInitializerDataset
from cesnet_tszoo.data_models.load_dataset_configs.time_load_config import TimeLoadConfig
from cesnet_tszoo.data_models.init_dataset_configs.time_init_config import TimeDatasetInitConfig
import cesnet_tszoo.datasets.utils.loaders as dataset_loaders
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME


@dataclass
class TimeBasedCesnetDataset(CesnetDataset):
    """This class is used for time-based returning of data. Can be created by using [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset] with parameter `dataset_type` = `DatasetType.TIME_BASED`.

    Time-based means batch size affects number of returned times in one batch and all sets have the same time series. Which time series are returned does not change. Additionally it supports sliding window.

    The dataset provides multiple ways to access the data:

    - **Iterable PyTorch DataLoader**: For batch processing.
    - **Pandas DataFrame**: For loading the entire training, validation, test or all set at once.
    - **Numpy array**: For loading the entire training, validation, test or all set at once. 
    - See [loading data][loading-data] for more details.

    The dataset is stored in a [PyTables](https://www.pytables.org/) database. The internal `TimeBasedDataset`, `SplittedDataset`, `TimeBasedInitializerDataset` classes (used only when calling [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]) act as wrappers that implement the PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 
    interface. These wrappers are compatible with PyTorch’s [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), providing efficient parallel data loading. 

    The dataset configuration is done through the [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig] class.       

    **Intended usage:**

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy].  

    Alternatively you can use [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark]

    1. Call [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark] with the desired benchmark. You can use your own saved benchmark or you can use already built-in one. This will download the dataset and annotations (if available) if they have not been previously downloaded.
    2. Retrieve the initialized dataset using [`get_initialized_dataset`][cesnet_tszoo.benchmarks.Benchmark.get_initialized_dataset]. This will provide a dataset that is ready to use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy].  

    Parameters:
        metadata: Holds various metadata used in dataset for its creation, loading data and similar.

    Attributes:
        metadata: Holds various metadata used in dataset for its creation, loading data and similar.
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
        all_dataset: All set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.
        train_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training set.
        val_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation set.
        test_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test set.         
        all_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for all set.        
    """

    dataset_config: Optional[TimeBasedConfig] = field(default=None, init=False)

    train_dataset: Optional[TimeBasedSplittedDataset] = field(default=None, init=False)
    val_dataset: Optional[TimeBasedSplittedDataset] = field(default=None, init=False)
    test_dataset: Optional[TimeBasedSplittedDataset] = field(default=None, init=False)
    all_dataset: Optional[TimeBasedSplittedDataset] = field(default=None, init=False)

    train_dataloader: Optional[TimeBasedDataloader] = field(default=None, init=False)
    val_dataloader: Optional[TimeBasedDataloader] = field(default=None, init=False)
    test_dataloader: Optional[TimeBasedDataloader] = field(default=None, init=False)
    all_dataloader: Optional[TimeBasedDataloader] = field(default=None, init=False)

    dataloader_factory: TimeBasedDataloaderFactory = field(default=TimeBasedDataloaderFactory(), init=False)

    dataset_type: DatasetType = field(default=DatasetType.TIME_BASED, init=False)

    _export_config_copy: Optional[TimeBasedConfig] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.logger.info("Dataset is time-based. Use cesnet_tszoo.configs.TimeBasedConfig")

    def set_dataset_config_and_initialize(self, dataset_config: TimeBasedConfig, display_config_details: bool = True, workers: int | Literal["config"] = "config") -> None:
        """
        Initialize training set, validation est, test set etc.. This method must be called before any data can be accessed. It is required for the final initialization of [`dataset_config`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig].

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

        assert dataset_config is not None, "Used dataset_config cannot be None."
        assert isinstance(dataset_config, TimeBasedConfig), f"This config is used for dataset of type '{dataset_config.dataset_type}'. Meanwhile this dataset is of type '{self.metadata.dataset_type}'."

        super(TimeBasedCesnetDataset, self).set_dataset_config_and_initialize(dataset_config, display_config_details, workers)

    def get_data_about_set(self, about: SplitType | Literal["train", "val", "test", "all"]) -> dict:
        """
        Retrieve data related to the specified set.

        Parameters:
            about: Specifies the set to retrieve data about.

        Returned dictionary contains:

        - **ts_ids:** Ids of time series in `about` set.
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
            if not self.dataset_config.has_train():
                raise ValueError("Train split is not used.")
            time_period = self.dataset_config.train_time_period
        elif about == SplitType.VAL:
            if not self.dataset_config.has_val():
                raise ValueError("Val split is not used.")
            time_period = self.dataset_config.val_time_period
        elif about == SplitType.TEST:
            if not self.dataset_config.has_test():
                raise ValueError("Test split is not used.")
            time_period = self.dataset_config.test_time_period
        elif about == SplitType.ALL:
            if not self.dataset_config.has_all():
                raise ValueError("All split is not used.")

            time_period = self.dataset_config.all_time_period
        else:
            raise ValueError("Invalid split type!")

        datetime_temp = np.array([datetime.fromtimestamp(time, timezone.utc) for time in self.metadata.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]]])

        result["ts_ids"] = self.dataset_config.ts_ids.copy()
        result[TimeFormat.ID_TIME] = time_period[ID_TIME_COLUMN_NAME].copy()
        result[TimeFormat.DATETIME] = datetime_temp.copy()
        result[TimeFormat.UNIX_TIME] = self.metadata.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]].copy()
        result[TimeFormat.SHIFTED_UNIX_TIME] = self.metadata.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]] - self.metadata.time_indices[TIME_COLUMN_NAME][0]

        return result

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
        """Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize], this method initializes the set datasets (train, validation, test and all). """

        if self.dataset_config.has_train():
            load_config = TimeLoadConfig(self.dataset_config, SplitType.TRAIN)
            self.train_dataset = TimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, load_config, self.dataset_config.train_workers)

            self.logger.debug("train_dataset initiliazed.")

        if self.dataset_config.has_val():
            load_config = TimeLoadConfig(self.dataset_config, SplitType.VAL)
            self.val_dataset = TimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, load_config, self.dataset_config.val_workers)

            self.logger.debug("val_dataset initiliazed.")

        if self.dataset_config.has_test():
            load_config = TimeLoadConfig(self.dataset_config, SplitType.TEST)
            self.test_dataset = TimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, load_config, self.dataset_config.test_workers)

            self.logger.debug("test_dataset initiliazed.")

        if self.dataset_config.has_all():
            load_config = TimeLoadConfig(self.dataset_config, SplitType.ALL)
            self.all_dataset = TimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, load_config, self.dataset_config.all_workers)

            self.logger.debug("all_dataset initiliazed.")

    def _initialize_transformers_and_details(self, workers: int) -> None:
        """
        Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]. 

        Goes through data to validate time series against `nan_threshold`, fit/partial fit `transformers`, `anomaly handlers` and prepare `fillers`.
        """

        init_config = TimeDatasetInitConfig(self.dataset_config)

        init_dataset = TimeBasedInitializerDataset(self.metadata.dataset_path, self.metadata.data_table_path, init_config)

        sampler = SequentialSampler(init_dataset)
        dataloader = DataLoader(init_dataset, num_workers=workers, collate_fn=self._collate_fn, worker_init_fn=TimeBasedInitializerDataset.worker_init_fn, persistent_workers=False, sampler=sampler)

        if workers == 0:
            init_dataset.pytables_worker_init()

        ts_ids_to_take = []

        self.logger.info("Updating config on train/val/test/all and selected time series.")
        for i, data in enumerate(tqdm(dataloader, total=len(self.dataset_config.ts_row_ranges))):
            train_data, train_count_values, val_count_values, test_count_values, all_count_values, val_filler, test_filler, anomaly_handler = data[0]

            missing_train_percentage = 0
            missing_val_percentage = 0
            missing_test_percentage = 0
            missing_all_percentage = 0

            # Filter time series based on missing data threshold
            if self.dataset_config.has_train():
                missing_train_percentage = train_count_values[1] / (train_count_values[0] + train_count_values[1])
            if self.dataset_config.has_val():
                missing_val_percentage = val_count_values[1] / (val_count_values[0] + val_count_values[1])
            if self.dataset_config.has_test():
                missing_test_percentage = test_count_values[1] / (test_count_values[0] + test_count_values[1])
            if self.dataset_config.has_all():
                missing_all_percentage = all_count_values[1] / (all_count_values[0] + all_count_values[1])

            if max(missing_train_percentage, missing_val_percentage, missing_test_percentage, missing_all_percentage) <= self.dataset_config.nan_threshold:
                ts_ids_to_take.append(i)

                # Fit transformers if required
                if train_data is not None and (not self.dataset_config.transformer_factory.has_already_initialized or self.dataset_config.partial_fit_initialized_transformers):

                    if self.dataset_config.transformer_factory.has_already_initialized and self.dataset_config.partial_fit_initialized_transformers:
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
                if self.dataset_config.has_val():
                    self.dataset_config.val_fillers[i] = val_filler
                if self.dataset_config.has_test():
                    self.dataset_config.test_fillers[i] = test_filler

                # Sets fitted anomaly handlers
                if self.dataset_config.has_train():
                    self.dataset_config.anomaly_handlers[i] = anomaly_handler

        if workers == 0:
            init_dataset.cleanup()

        if len(ts_ids_to_take) == 0:
            raise ValueError("No valid time series left in ts_ids after applying nan_threshold.")

        # Update config based on filtered time series
        self.dataset_config.ts_row_ranges = self.dataset_config.ts_row_ranges[ts_ids_to_take]
        self.dataset_config.ts_ids = self.dataset_config.ts_ids[ts_ids_to_take]

        if self.dataset_config.create_transformer_per_time_series:
            self.dataset_config.transformers = self.dataset_config.transformers[ts_ids_to_take]

        if self.dataset_config.has_train():
            self.dataset_config.train_fillers = self.dataset_config.train_fillers[ts_ids_to_take]
            self.dataset_config.anomaly_handlers = self.dataset_config.anomaly_handlers[ts_ids_to_take]
        if self.dataset_config.has_val():
            self.dataset_config.val_fillers = self.dataset_config.val_fillers[ts_ids_to_take]
        if self.dataset_config.has_test():
            self.dataset_config.test_fillers = self.dataset_config.test_fillers[ts_ids_to_take]
        if self.dataset_config.has_all():
            self.dataset_config.all_fillers = self.dataset_config.all_fillers[ts_ids_to_take]

        self.logger.debug("ts_ids updated: %s time series left.", len(ts_ids_to_take))

    def _update_export_config_copy(self) -> None:
        """
        Called at the end of [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize] or when changing config values. 

        Updates values of config used for saving config.
        """
        self._export_config_copy.database_name = self.metadata.database_name

        if self.dataset_config.ts_ids is not None:
            self._export_config_copy.ts_ids = self.dataset_config.ts_ids.copy()
            self.logger.debug("Updated ts_ids of _export_config_copy.")
        else:
            self._export_config_copy.ts_ids = None
            self.logger.debug("Updated ts_ids of _export_config_copy.")

        self._export_config_copy.sliding_window_size = self.dataset_config.sliding_window_size
        self._export_config_copy.sliding_window_prediction_size = self.dataset_config.sliding_window_prediction_size
        self._export_config_copy.sliding_window_step = self.dataset_config.sliding_window_step
        self._export_config_copy.set_shared_size = self.dataset_config.set_shared_size

        super(TimeBasedCesnetDataset, self)._update_export_config_copy()

    def _get_singular_time_series_dataset(self, parent_dataset: TimeBasedSplittedDataset, ts_id: int) -> TimeBasedSplittedDataset:
        """Returns dataset for single time series """

        temp = np.where(np.isin(parent_dataset.load_config.ts_row_ranges[self.metadata.ts_id_name], [ts_id]))[0]

        if len(temp) == 0:
            raise ValueError(f"ts_id {ts_id} was not found in valid time series for this set. Available time series are: {parent_dataset.ts_row_ranges[self.metadata.ts_id_name]}")

        time_series_position = temp[0]

        split_load_config = parent_dataset.load_config.create_split_copy(slice(time_series_position, time_series_position + 1))

        dataset = TimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, split_load_config, 0)
        self.logger.debug("Singular time series dataset initiliazed.")

        return dataset

    def _get_data_for_plot(self, ts_id: int, feature_indices: np.ndarray[int], time_format: TimeFormat) -> tuple[np.ndarray, np.ndarray]:
        """Dataset type specific retrieval of data. """

        # Validate the time series ID (ts_id)
        id_result = np.argwhere(np.isin(self.dataset_config.ts_ids, ts_id)).ravel()

        if len(id_result) == 0:
            raise ValueError(f"Invalid ts_id '{ts_id}'. The provided ts_id is not found in the available time series IDs.", self.dataset_config.ts_ids)
        else:
            id_result = id_result[0]
            self.logger.debug("Valid ts_id found: %d", id_result)

        data = None

        if self.dataset_config.has_train():
            data = self.__update_data_for_plot(self.train_dataset, ts_id, feature_indices, data)

        if self.dataset_config.has_val():
            data = self.__update_data_for_plot(self.val_dataset, ts_id, feature_indices, data)

        if self.dataset_config.has_test():
            data = self.__update_data_for_plot(self.test_dataset, ts_id, feature_indices, data)

        return data, self.get_data_about_set(SplitType.ALL)[time_format]

    def __update_data_for_plot(self, dataset: TimeBasedSplittedDataset, ts_id: int, feature_indices: list[int], previous_data: Optional[np.ndarray]):
        dataset = self._get_singular_time_series_dataset(dataset, ts_id)

        dataloader = self.dataloader_factory.create_dataloader(dataset, self.dataset_config, 0, True, None)

        temp_data = dataset_loaders.create_numpy_from_dataloader(dataloader, np.array([ts_id]), dataset.load_config.time_format, dataset.load_config.include_time, DatasetType.TIME_BASED, True)

        if (dataset.load_config.time_format == TimeFormat.DATETIME and dataset.load_config.include_time):
            temp_data = temp_data[0]

        temp_data = temp_data[0][:, feature_indices]

        if previous_data is None:
            return temp_data

        return np.concatenate([previous_data, temp_data])
