from datetime import datetime, timezone
from typing import Optional, Literal
from numbers import Number
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from torch.utils.data import DataLoader, SequentialSampler

from cesnet_tszoo.utils.enums import SplitType, TimeFormat, DatasetType, FillerType, TransformerType, AnomalyHandlerType, PreprocessType
from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.configs.time_based_config import TimeBasedConfig
from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.configs.config_editors.time_based_config_editor import TimeBasedConfigEditor
from cesnet_tszoo.pytables_data.datasets.time_based import TimeBasedDataloader, TimeBasedDataloaderFactory
from cesnet_tszoo.pytables_data.time_based_initializer_dataset import TimeBasedInitializerDataset
from cesnet_tszoo.pytables_data.time_based_splitted_dataset import TimeBasedSplittedDataset
from cesnet_tszoo.data_models.load_dataset_configs.time_load_config import TimeLoadConfig
from cesnet_tszoo.data_models.init_dataset_configs.time_init_config import TimeDatasetInitConfig
from cesnet_tszoo.data_models.init_dataset_return import InitDatasetReturn
from cesnet_tszoo.data_models.preprocess_order_group import PreprocessOrderGroup
import cesnet_tszoo.datasets.utils.loaders as dataset_loaders
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
import cesnet_tszoo.utils.css_styles.utils as css_utils


@dataclass
class TimeBasedCesnetDataset(CesnetDataset):
    """This class is used for time-based returning of data. Can be created by using [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset) with parameter `dataset_type` = `DatasetType.TIME_BASED`.

    Time-based means batch size affects number of returned times in one batch and all sets have the same time series. Which time series are returned does not change. Additionally it supports sliding window.

    The dataset provides multiple ways to access the data:

    - **Iterable PyTorch DataLoader**: For batch processing.
    - **Pandas DataFrame**: For loading the entire training, validation, test or all set at once.
    - **Numpy array**: For loading the entire training, validation, test or all set at once. 
    - See [loading data][loading-data] for more details.

    The dataset is stored in a [PyTables](https://www.pytables.org/) dataset. The internal `TimeBasedSplittedDataset`, `TimeSplitBasedDataset` and `TimeBasedInitializerDataset` classes act as wrappers that implement the PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 
    interface. These wrappers are compatible with PyTorch’s [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), providing efficient parallel data loading. 

    The dataset configuration is done through the [`TimeBasedConfig`](reference_time_based_config.md#references.TimeBasedConfig) class.       

    **Intended usage:**

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset). This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`TimeBasedConfig`](reference_time_based_config.md#references.TimeBasedConfig) and set it using [`set_dataset_config_and_initialize`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize). 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset)/[`get_train_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df)/[`get_train_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy) to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader)/[`get_val_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df)/[`get_val_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy).
    5. Evaluate the model on [`get_test_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader)/[`get_test_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df)/[`get_test_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy).  

    Alternatively you can use [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark]

    1. Call [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark] with the desired benchmark. You can use your own saved benchmark or you can use already built-in one. This will download the dataset and annotations (if available) if they have not been previously downloaded.
    2. Retrieve the initialized dataset using [`get_initialized_dataset`](reference_benchmarks.md#cesnet_tszoo.benchmarks.Benchmark.get_initialized_dataset). This will provide a dataset that is ready to use.
    3. Use [`get_train_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset)/[`get_train_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df)/[`get_train_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy) to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader)/[`get_val_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df)/[`get_val_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy).
    5. Evaluate the model on [`get_test_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader)/[`get_test_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df)/[`get_test_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy).  
    """

    dataset_config: Optional[TimeBasedConfig] = field(default=None, init=False)
    """Configuration of the dataset."""

    train_dataset: Optional[TimeBasedSplittedDataset] = field(default=None, init=False)
    """Training set as a `TimeBasedSplittedDataset` instance wrapping multiple `TimeSplitBasedDataset` that wrap the PyTables dataset."""

    val_dataset: Optional[TimeBasedSplittedDataset] = field(default=None, init=False)
    """Validation set as a `TimeBasedSplittedDataset` instance wrapping multiple `TimeSplitBasedDataset` that wrap the PyTables dataset."""

    test_dataset: Optional[TimeBasedSplittedDataset] = field(default=None, init=False)
    """Test set as a `TimeBasedSplittedDataset` instance wrapping multiple `TimeSplitBasedDataset` that wrap the PyTables dataset.  """

    all_dataset: Optional[TimeBasedSplittedDataset] = field(default=None, init=False)
    """All set as a `TimeBasedSplittedDataset` instance wrapping multiple `TimeSplitBasedDataset` that wrap the PyTables dataset.   """

    train_dataloader: Optional[TimeBasedDataloader] = field(default=None, init=False)
    """Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training set."""
    val_dataloader: Optional[TimeBasedDataloader] = field(default=None, init=False)
    """Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation set."""
    test_dataloader: Optional[TimeBasedDataloader] = field(default=None, init=False)
    """Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test set.   """
    all_dataloader: Optional[TimeBasedDataloader] = field(default=None, init=False)
    """Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for all set.   """

    dataloader_factory: TimeBasedDataloaderFactory = field(default=TimeBasedDataloaderFactory(), init=False)
    """Factory used to create TimeBasedDataloader.  """

    dataset_type: DatasetType = field(default=DatasetType.TIME_BASED, init=False)

    _export_config_copy: Optional[TimeBasedConfig] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.logger.info("Dataset is time-based. Use cesnet_tszoo.configs.TimeBasedConfig")

    def set_dataset_config_and_initialize(self, dataset_config: TimeBasedConfig, display_config_details: Optional[Literal["text", "diagram"]] = "text", workers: int | Literal["config"] = "config") -> None:
        """
        Initialize training set, validation set, test set etc.. This method must be called before any data can be accessed. It is required for the final initialization of [`dataset_config`][references.TimeBasedConfig].

        The following configuration attributes are used during initialization:

        Dataset config | Description
        -------------- | -----------
        `init_workers` | Specifies the number of workers to use for initialization. Applied when `workers` = "config".
        `partial_fit_initialized_transformers` | Determines whether initialized transformers should be partially fitted on the training data.
        `nan_threshold` | Filters out time series with missing values exceeding the specified threshold.

        Parameters:
            dataset_config: Desired configuration of the dataset.
            display_config_details: Flag indicating whether and how to display the configuration values after initialization. `Default: text`  
            workers: The number of workers to use during initialization. `Default: "config"`  
        """

        assert dataset_config is not None, "Used dataset_config cannot be None."
        assert isinstance(dataset_config, TimeBasedConfig), f"This config is used for dataset of type '{dataset_config.dataset_type}'. Meanwhile this dataset is of type '{self.metadata.dataset_type}'."

        super(TimeBasedCesnetDataset, self).set_dataset_config_and_initialize(dataset_config, display_config_details, workers)

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
                                             preprocess_order: list[str, type] | Literal["config"] = "config",
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
                                             display_config_details: Optional[Literal["text", "diagram"]] = None):
        """Used for updating selected configurations set in config.
        Set parameter to `config` to keep it as it is config.
        If exception is thrown during set, no changes are made.

        Can affect following configuration:

        Dataset config | Description
        -------------- | -----------
        `default_values` | Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature.
        `sliding_window_size` | Number of times in one window. Impacts dataloader behavior. Refer to relevant config for details.
        `sliding_window_prediction_size` | Number of times to predict from sliding_window_size. Refer to relevant config for details.
        `sliding_window_step` | Number of times to move by after each window. Refer to relevant config for details.
        `set_shared_size` | How much times should time periods share. Order of sharing is training set < validation set < test set. Refer to relevant config for details.
        `train_batch_size` | Number of samples per batch for train set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.
        `val_batch_size` | Number of samples per batch for val set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.
        `test_batch_size` | Number of samples per batch for test set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.
        `all_batch_size` | Number of samples per batch for all set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.
        `preprocess_order` | Used order of when preprocesses are applied. Can be also used to add/remove custom handlers.
        `fill_missing_with` | Defines how to fill missing values in the dataset.
        `transform_with` | Defines the transformer to transform the dataset.
        `handle_anomalies_with` | Defines the anomaly handler to handle anomalies in the dataset.
        `create_transformer_per_time_series` | If `True`, a separate transformer is created for each time series. Not used when using already initialized transformers.
        `partial_fit_initialized_transformers` | If `True`, partial fitting on train set is performed when using initialized transformers.
        `train_workers` | Number of workers for loading training data.
        `val_workers` | Number of workers for loading validation data.
        `test_workers` | Number of workers for loading test data.
        `all_workers` | Number of workers for loading all data.
        `init_workers` | Number of workers for dataset configuration.


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
            preprocess_order: Used order of when preprocesses are applied. Can be also used to add/remove custom handlers. `Defaults: config`.               
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

        config_editor = TimeBasedConfigEditor(self._export_config_copy,
                                              default_values,
                                              train_batch_size,
                                              val_batch_size,
                                              test_batch_size,
                                              all_batch_size,
                                              preprocess_order,
                                              fill_missing_with,
                                              transform_with,
                                              handle_anomalies_with,
                                              create_transformer_per_time_series,
                                              partial_fit_initialized_transformers,
                                              train_workers,
                                              val_workers,
                                              test_workers,
                                              all_workers,
                                              init_workers,
                                              sliding_window_size,
                                              sliding_window_prediction_size,
                                              sliding_window_step,
                                              set_shared_size
                                              )

        self._update_dataset_config_and_initialize(config_editor, workers, display_config_details)

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

        Affects following configuration:

        Dataset config | Description
        -------------- | -----------
        `sliding_window_size` | Number of times in one window. Impacts dataloader behavior. Refer to relevant config for details.
        `sliding_window_prediction_size` | Number of times to predict from sliding_window_size. Refer to relevant config for details.
        `sliding_window_step` | Number of times to move by after each window. Refer to relevant config for details.
        `set_shared_size` | How much times should time periods share. Order of sharing is training set < validation set < test set. Refer to relevant config for details.

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
        """Called in [`set_dataset_config_and_initialize`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize), this method initializes the set datasets (train, validation, test and all). """

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
        Called in [`set_dataset_config_and_initialize`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize). 

        Goes through data to validate time series against `nan_threshold`, fit/partial fit `transformers`, `anomaly handlers` and prepare `fillers`.
        """

        self.logger.info("Updating config on train/val/test/all and selected time series.")

        is_first_cycle = True

        train_groups = self.dataset_config._get_train_preprocess_init_order_groups()
        val_groups = self.dataset_config._get_val_preprocess_init_order_groups()
        test_groups = self.dataset_config._get_test_preprocess_init_order_groups()

        grouped = list(zip(train_groups, val_groups, test_groups))
        ts_ids_ignore = np.zeros_like(self.dataset_config.ts_row_ranges, dtype=np.bool)
        ts_ids_to_take = []

        for i, groups in enumerate(grouped):
            ts_ids_to_take = []
            train_group, val_group, test_group = groups

            self.logger.info("Starting fitting cycle %s/%s.", i + 1, len(grouped))

            init_config = TimeDatasetInitConfig(self.dataset_config, ts_ids_ignore, train_group, val_group, test_group)
            init_dataset = TimeBasedInitializerDataset(self.metadata.dataset_path, self.metadata.data_table_path, init_config)

            sampler = SequentialSampler(init_dataset)
            dataloader = DataLoader(init_dataset, num_workers=workers, collate_fn=self._collate_fn, worker_init_fn=TimeBasedInitializerDataset.worker_init_fn, persistent_workers=False, sampler=sampler)

            if workers == 0:
                init_dataset.pytables_worker_init()

            for ts_id, data in enumerate(tqdm(dataloader, total=len(self.dataset_config.ts_row_ranges))):

                if ts_ids_ignore[ts_id]:
                    continue

                train_return: InitDatasetReturn
                val_return: InitDatasetReturn
                test_return: InitDatasetReturn
                all_return: InitDatasetReturn
                train_return, val_return, test_return, all_return = data[0]

                if train_return.is_under_nan_threshold and val_return.is_under_nan_threshold and test_return.is_under_nan_threshold and all_return.is_under_nan_threshold:
                    ts_ids_to_take.append(ts_id)

                    if self.dataset_config.has_train():
                        self.__update_based_on_train_init_return(train_return, train_group, ts_id)

                    if self.dataset_config.has_val() or self.dataset_config.has_test():
                        self.__update_based_on_non_fit_returns(val_return, test_return, val_group, test_group, ts_id)

            if workers == 0:
                init_dataset.cleanup()

            if is_first_cycle:

                if len(ts_ids_to_take) == 0:
                    raise ValueError("No valid time series left in ts_ids after applying nan_threshold.")

                ts_ids_ignore = np.ones_like(self.dataset_config.ts_row_ranges, dtype=np.bool)
                ts_ids_ignore[ts_ids_to_take] = False
                self.logger.debug("invalid ts_ids flagged: %s time series left.", len(ts_ids_to_take))

                is_first_cycle = False

        # Update config based on filtered time series
        self.dataset_config.ts_row_ranges = self.dataset_config.ts_row_ranges[ts_ids_to_take]
        self.dataset_config.ts_ids = self.dataset_config.ts_ids[ts_ids_to_take]

        if self.dataset_config.has_train():
            self.dataset_config._update_preprocess_order_supported_ids(self.dataset_config.train_preprocess_order, ts_ids_to_take)
        if self.dataset_config.has_val():
            self.dataset_config._update_preprocess_order_supported_ids(self.dataset_config.val_preprocess_order, ts_ids_to_take)
        if self.dataset_config.has_test():
            self.dataset_config._update_preprocess_order_supported_ids(self.dataset_config.test_preprocess_order, ts_ids_to_take)
        if self.dataset_config.has_all():
            self.dataset_config._update_preprocess_order_supported_ids(self.dataset_config.all_preprocess_order, ts_ids_to_take)

    def __update_based_on_train_init_return(self, train_return: InitDatasetReturn, train_group: PreprocessOrderGroup, ts_id: int):
        fitted_inner_index = 0
        for inner_preprocess_order in train_group.preprocess_inner_orders:
            if inner_preprocess_order.should_be_fitted:
                inner_preprocess_order.holder.update_instance(train_return.preprocess_fitted_instances[fitted_inner_index].instance, ts_id)
                fitted_inner_index += 1

        # updates outer preprocessors based on passed train data from InitDataset
        for outer_preprocess_order in train_group.preprocess_outer_orders:
            if outer_preprocess_order.should_be_fitted:
                outer_preprocess_order.holder.fit(train_return.train_data, ts_id)

            if outer_preprocess_order.can_be_applied:
                train_return.train_data = outer_preprocess_order.holder.apply(train_return.train_data, ts_id)

    def __update_based_on_non_fit_returns(self, val_return: InitDatasetReturn, test_return: InitDatasetReturn, val_group: PreprocessOrderGroup, test_group: PreprocessOrderGroup, ts_id: int):

        if self.dataset_config.has_val():
            fitted_inner_index = 0
            for inner_preprocess_order in val_group.preprocess_inner_orders:
                if inner_preprocess_order.should_be_fitted:
                    inner_preprocess_order.holder.update_instance(val_return.preprocess_fitted_instances[fitted_inner_index].instance, ts_id)
                    fitted_inner_index += 1

        if self.dataset_config.has_test():
            fitted_inner_index = 0
            for inner_preprocess_order in test_group.preprocess_inner_orders:
                if inner_preprocess_order.should_be_fitted:
                    inner_preprocess_order.holder.update_instance(test_return.preprocess_fitted_instances[fitted_inner_index].instance, ts_id)
                    fitted_inner_index += 1

    def _update_export_config_copy(self) -> None:
        """
        Called at the end of [`set_dataset_config_and_initialize`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize) or when changing config values. 

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
