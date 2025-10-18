from datetime import datetime, timezone
from typing import Optional, Literal
from dataclasses import dataclass, field
from numbers import Number

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from cesnet_tszoo.utils.enums import SplitType, TimeFormat, DatasetType, TransformerType, FillerType, AnomalyHandlerType, PreprocessType
from cesnet_tszoo.configs.disjoint_time_based_config import DisjointTimeBasedConfig
from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.configs.config_editors.disjoint_time_based_config_editor import DisjointTimeBasedConfigEditor
from cesnet_tszoo.pytables_data.datasets.disjoint_time_based import DisjointTimeBasedDataloaderFactory, DisjointTimeBasedDataloader
from cesnet_tszoo.pytables_data.disjoint_based_initializer_dataset import DisjointTimeBasedInitializerDataset
from cesnet_tszoo.pytables_data.disjoint_based_splitted_dataset import DisjointTimeBasedSplittedDataset
import cesnet_tszoo.datasets.utils.loaders as dataset_loaders
from cesnet_tszoo.data_models.init_dataset_configs.disjoint_time_init_config import DisjointTimeDatasetInitConfig
from cesnet_tszoo.data_models.load_dataset_configs.disjoint_time_load_config import DisjointTimeLoadConfig
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
from cesnet_tszoo.data_models.init_dataset_return import InitDatasetReturn
from cesnet_tszoo.data_models.preprocess_order_group import PreprocessOrderGroup


@dataclass
class DisjointTimeBasedCesnetDataset(CesnetDataset):
    """This class is used for disjoint-time-based returning of data. Can be created by using [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset] with parameter `dataset_type` = `DatasetType.DISJOINT_TIME_BASED`.

    Disjoint-time-based means batch size affects number of returned times in one batch and each set can have different time series. Which time series are returned does not change. Additionally it supports sliding window.

    The dataset provides multiple ways to access the data:

    - **Iterable PyTorch DataLoader**: For batch processing.
    - **Pandas DataFrame**: For loading the entire training, validation or test set at once.
    - **Numpy array**: For loading the entire training, validation or test set at once. 
    - See [loading data][loading-data] for more details.

    The dataset is stored in a [PyTables](https://www.pytables.org/) database. The internal `TimeBasedDataset`, `SplittedDataset`, `TimeBasedInitializerDataset` classes (used only when calling [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize]) act as wrappers that implement the PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 
    interface. These wrappers are compatible with PyTorch’s [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), providing efficient parallel data loading. 

    The dataset configuration is done through the [`DisjointTimeBasedConfig`][cesnet_tszoo.configs.disjoint_time_based_config.DisjointTimeBasedConfig] class.       

    **Intended usage:**

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`DisjointTimeBasedConfig`][cesnet_tszoo.configs.disjoint_time_based_config.DisjointTimeBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_numpy].  

    Alternatively you can use [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark]

    1. Call [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark] with the desired benchmark. You can use your own saved benchmark or you can use already built-in one. This will download the dataset and annotations (if available) if they have not been previously downloaded.
    2. Retrieve the initialized dataset using [`get_initialized_dataset`][cesnet_tszoo.benchmarks.Benchmark.get_initialized_dataset]. This will provide a dataset that is ready to use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_numpy].  

    Parameters:
        metadata: Holds various metadata used in dataset for its creation, loading data and similar.

    Attributes:
        metadata: Holds various metadata used in dataset for its creation, loading data and similar.
        annotations: Annotations for the selected dataset.
        logger: Logger for displaying information.  
        imported_annotations_ts_identifier: Identifier for the imported annotations of type `AnnotationType.TS_ID`.
        imported_annotations_time_identifier: Identifier for the imported annotations of type `AnnotationType.ID_TIME`.
        imported_annotations_both_identifier: Identifier for the imported annotations of type `AnnotationType.BOTH`.  

    The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize] is called.

    Attributes:
        dataset_config: Configuration of the dataset.
        train_dataset: Training set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.
        val_dataset: Validation set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.
        test_dataset: Test set as a `SplittedDataset` instance wrapping multiple `TimeBasedDataset` that wrap the PyTables database.  
        train_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training set.
        val_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation set.
        test_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test set.              
    """

    dataset_config: Optional[DisjointTimeBasedConfig] = field(default=None, init=False)

    train_dataset: Optional[DisjointTimeBasedSplittedDataset] = field(default=None, init=False)
    val_dataset: Optional[DisjointTimeBasedSplittedDataset] = field(default=None, init=False)
    test_dataset: Optional[DisjointTimeBasedSplittedDataset] = field(default=None, init=False)

    train_dataloader: Optional[DisjointTimeBasedDataloader] = field(default=None, init=False)
    val_dataloader: Optional[DisjointTimeBasedDataloader] = field(default=None, init=False)
    test_dataloader: Optional[DisjointTimeBasedDataloader] = field(default=None, init=False)

    dataloader_factory: DisjointTimeBasedDataloaderFactory = field(default=DisjointTimeBasedDataloaderFactory(), init=False)

    dataset_type: DatasetType = field(default=DatasetType.DISJOINT_TIME_BASED, init=False)

    _export_config_copy: Optional[DisjointTimeBasedConfig] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.logger.info("Dataset is disjoint_time_based. Use cesnet_tszoo.configs.DisjointTimeBasedConfig")

    def set_dataset_config_and_initialize(self, dataset_config: DisjointTimeBasedConfig, display_config_details: bool = True, workers: int | Literal["config"] = "config") -> None:
        """
        Initialize training set, validation est, test set etc.. This method must be called before any data can be accessed. It is required for the final initialization of [`dataset_config`][cesnet_tszoo.configs.disjoint_time_based_config.DisjointTimeBasedConfig].

        The following configuration attributes are used during initialization:

        | Dataset config                              | Description                                                                                    |
        | ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
        | `init_workers`                              | Specifies the number of workers to use for initialization. Applied when `workers` = "config".  |
        | `partial_fit_initialized_transformers`      | Determines whether initialized transformers should be partially fitted on the training data.   |
        | `nan_threshold`                             | Filters out time series with missing values exceeding the specified threshold.                 |

        Parameters:
            dataset_config: Desired configuration of the dataset.
            display_config_details: Flag indicating whether to display the configuration values after initialization. `Default: True`  
            workers: The number of workers to use during initialization. `Default: "config"`  
        """

        assert dataset_config is not None, "Used dataset_config cannot be None."
        assert isinstance(dataset_config, DisjointTimeBasedConfig), f"This config is used for dataset of type '{dataset_config.dataset_type}'. Meanwhile this dataset is of type '{self.metadata.dataset_type}'."

        super(DisjointTimeBasedCesnetDataset, self).set_dataset_config_and_initialize(dataset_config, display_config_details, workers)

    def apply_transformer(self, transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "l2_normalizer"] | None | Literal["config"] = "config",
                          partial_fit_initialized_transformers: bool | Literal["config"] = "config", workers: int | Literal["config"] = "config") -> None:
        """Used for updating transformer and relevenat configurations set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                         | Description                                                                                                    |
        | -------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
        | `transform_with`                       | Defines the transformer to transform the dataset.                                                              |     
        | `partial_fit_initialized_transformers` | If `True`, partial fitting on train set is performed when using initiliazed transformers.                      |    

        Parameters:
            transform_with: Defines the transformer to transform the dataset. `Defaults: config`.  
            partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers. `Defaults: config`.  
            workers: How many workers to use when setting new transformer. `Defaults: config`.      
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating transformer values.")

        self.update_dataset_config_and_initialize(transform_with=transform_with, partial_fit_initialized_transformers=partial_fit_initialized_transformers, workers=workers)

    def update_dataset_config_and_initialize(self,
                                             default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None | Literal["config"] = "config",
                                             sliding_window_size: int | None | Literal["config"] = "config",
                                             sliding_window_prediction_size: int | None | Literal["config"] = "config",
                                             sliding_window_step: int | Literal["config"] = "config",
                                             set_shared_size: float | int | Literal["config"] = "config",
                                             train_batch_size: int | Literal["config"] = "config",
                                             val_batch_size: int | Literal["config"] = "config",
                                             test_batch_size: int | Literal["config"] = "config",
                                             fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None | Literal["config"] = "config",
                                             transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "l2_normalizer"] | None | Literal["config"] = "config",
                                             handle_anomalies_with: type | AnomalyHandlerType | Literal["z-score", "interquartile_range"] | None | Literal["config"] = "config",
                                             partial_fit_initialized_transformers: bool | Literal["config"] = "config",
                                             train_workers: int | Literal["config"] = "config",
                                             val_workers: int | Literal["config"] = "config",
                                             test_workers: int | Literal["config"] = "config",
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
        | `fill_missing_with`                     | Defines how to fill missing values in the dataset.                                                                                              |     
        | `transform_with`                        | Defines the transformer to transform the dataset.                                                                                               |
        | `handle_anomalies_with`                 | Defines the anomaly handler to handle anomalies in the train set.                                                                               |             
        | `partial_fit_initialized_transformers`  | If `True`, partial fitting on train set is performed when using initiliazed transformers.                                                       |   
        | `train_workers`                         | Number of workers for loading training data.                                                                                                    |
        | `val_workers`                           | Number of workers for loading validation data.                                                                                                  |
        | `test_workers`                          | Number of workers for loading test data.                                                                                                        |  
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
            fill_missing_with: Defines how to fill missing values in the dataset. `Defaults: config`. 
            transform_with: Defines the transformer to transform the dataset. `Defaults: config`. 
            handle_anomalies_with: Defines the anomaly handler to handle anomalies in the train set. `Defaults: config`. 
            partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers. `Defaults: config`.    
            train_workers: Number of workers for loading training data. `Defaults: config`.
            val_workers: Number of workers for loading validation data. `Defaults: config`.
            test_workers: Number of workers for loading test data. `Defaults: config`.
            init_workers: Number of workers for dataset configuration. `Defaults: config`.                          
            workers: How many workers to use when updating configuration. `Defaults: config`.  
            display_config_details: Whether config details should be displayed after configuration. `Defaults: False`. 
        """

        config_editor = DisjointTimeBasedConfigEditor(self._export_config_copy,
                                                      default_values,
                                                      train_batch_size,
                                                      val_batch_size,
                                                      test_batch_size,
                                                      fill_missing_with,
                                                      transform_with,
                                                      handle_anomalies_with,
                                                      "config",
                                                      partial_fit_initialized_transformers,
                                                      train_workers,
                                                      val_workers,
                                                      test_workers,
                                                      init_workers,
                                                      sliding_window_size,
                                                      sliding_window_prediction_size,
                                                      sliding_window_step,
                                                      set_shared_size
                                                      )

        self._update_dataset_config_and_initialize(config_editor, workers, display_config_details)

    def get_data_about_set(self, about: SplitType | Literal["train", "val", "test"]) -> dict:
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
        time_series = None

        result = {}

        if about == SplitType.TRAIN:
            if not self.dataset_config.has_train():
                raise ValueError("Train split is not used.")
            time_period = self.dataset_config.train_time_period
            time_series = self.dataset_config.train_ts
        elif about == SplitType.VAL:
            if not self.dataset_config.has_val():
                raise ValueError("Val split is not used.")
            time_period = self.dataset_config.val_time_period
            time_series = self.dataset_config.val_ts
        elif about == SplitType.TEST:
            if not self.dataset_config.has_test():
                raise ValueError("Test split is not used.")
            time_period = self.dataset_config.test_time_period
            time_series = self.dataset_config.test_ts
        else:
            raise ValueError("Specified about parameter is not supported.")

        datetime_temp = np.array([datetime.fromtimestamp(time, timezone.utc) for time in self.metadata.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]]])

        result["ts_ids"] = time_series.copy()
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

    def set_batch_sizes(self, train_batch_size: int | Literal["config"] = "config", val_batch_size: int | Literal["config"] = "config", test_batch_size: int | Literal["config"] = "config") -> None:
        """Used for updating batch sizes set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                    | Description                                                                                                                                     |
        | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
        | `train_batch_size`                | Number of samples per batch for train set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details. |
        | `val_batch_size`                  | Number of samples per batch for val set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.   |
        | `test_batch_size`                 | Number of samples per batch for test set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.  |      

        Parameters:
            train_batch_size: Number of samples per batch for train set. `Defaults: config`.
            val_batch_size: Number of samples per batch for val set. `Defaults: config`.
            test_batch_size: Number of samples per batch for test set. `Defaults: config`.
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating batch sizes.")

        self.update_dataset_config_and_initialize(train_batch_size=train_batch_size, val_batch_size=val_batch_size, test_batch_size=test_batch_size, workers="config")
        self.logger.info("Batch sizes has been changed successfuly.")

    def set_workers(self, train_workers: int | Literal["config"] = "config", val_workers: int | Literal["config"] = "config",
                    test_workers: int | Literal["config"] = "config", init_workers: int | Literal["config"] = "config") -> None:
        """Used for updating workers set in config.

        Set parameter to `config` to keep it as it is config.

        If exception is thrown during set, no changes are made.

        Affects following configuration. 

        | Dataset config                 | Description                                    |
        | ------------------------------ | ---------------------------------------------- |
        | `train_workers`                | Number of workers for loading training data.   |
        | `val_workers`                  | Number of workers for loading validation data. |
        | `test_workers`                 | Number of workers for loading test data.       | 
        | `init_workers`                 | Number of workers for dataset configuration.   |      

        Parameters:
            train_workers: Number of workers for loading training data. `Defaults: config`.
            val_workers: Number of workers for loading validation data. `Defaults: config`.
            test_workers: Number of workers for loading test data. `Defaults: config`.
            init_workers: Number of workers for dataset configuration. `Defaults: config`.            
        """

        if self.dataset_config is None or not self.dataset_config.is_initialized:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before updating workers.")

        self.update_dataset_config_and_initialize(train_workers=train_workers, val_workers=val_workers, test_workers=test_workers, init_workers=init_workers, workers="config")
        self.logger.info("Workers has been changed successfuly.")

    def _initialize_datasets(self) -> None:
        """Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize], this method initializes the set datasets (train, validation, test and all). """

        if self.dataset_config.has_train():
            load_config = DisjointTimeLoadConfig(self.dataset_config, SplitType.TRAIN)
            self.train_dataset = DisjointTimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, load_config, self.dataset_config.train_workers)

            self.logger.debug("train_dataset initiliazed.")

        if self.dataset_config.has_val():
            load_config = DisjointTimeLoadConfig(self.dataset_config, SplitType.VAL)
            self.val_dataset = DisjointTimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, load_config, self.dataset_config.val_workers)

            self.logger.debug("val_dataset initiliazed.")

        if self.dataset_config.has_test():
            load_config = DisjointTimeLoadConfig(self.dataset_config, SplitType.TEST)
            self.test_dataset = DisjointTimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, load_config, self.dataset_config.test_workers)
            self.logger.debug("test_dataset initiliazed.")

    def _initialize_transformers_and_details(self, workers: int) -> None:
        """
        Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize]. 

        Goes through data to validate time series against `nan_threshold`, fit/partial fit `transformers`, fit `anomaly handlers` and prepare `fillers`.
        """

        if self.dataset_config.has_train():

            self.__initialize_config_for_train_set(workers)

            self.logger.debug("Train set updated: %s time series left.", len(self.dataset_config.train_ts))

        if self.dataset_config.has_val():
            init_config = DisjointTimeDatasetInitConfig(self.dataset_config, SplitType.VAL, PreprocessOrderGroup([]))

            ts_ids_to_take = self.__initialize_config_for_non_fit_sets(init_config, workers, "val")
            self.dataset_config.val_ts = self.dataset_config.val_ts[ts_ids_to_take]
            self.dataset_config.val_ts_row_ranges = self.dataset_config.val_ts_row_ranges[ts_ids_to_take]
            self.dataset_config.val_fillers = self.dataset_config.val_fillers[ts_ids_to_take]

            self.logger.debug("Val set updated: %s time series left.", len(self.dataset_config.val_ts))

        if self.dataset_config.has_test():
            init_config = DisjointTimeDatasetInitConfig(self.dataset_config, SplitType.TEST, PreprocessOrderGroup([]))

            ts_ids_to_take = self.__initialize_config_for_non_fit_sets(init_config, workers, "test")
            self.dataset_config.test_ts = self.dataset_config.test_ts[ts_ids_to_take]
            self.dataset_config.test_ts_row_ranges = self.dataset_config.test_ts_row_ranges[ts_ids_to_take]
            self.dataset_config.test_fillers = self.dataset_config.test_fillers[ts_ids_to_take]

            self.logger.debug("Test set updated: %s time series left.", len(self.dataset_config.test_ts))

        self.logger.info("Dataset initialization complete. Configuration updated.")

    def __initialize_config_for_train_set(self, workers: int) -> None:
        """Initializes config for provided time series. """

        self.logger.info("Updating config for train set and fitting values.")

        is_first_cycle = True

        groups = self.dataset_config._get_train_preprocess_init_order_groups()
        for ts_id, group in enumerate(groups):
            self.logger.info("Starting fitting cycle %s/%s.", ts_id + 1, len(groups))

            init_config = DisjointTimeDatasetInitConfig(self.dataset_config, SplitType.TRAIN, group)
            init_dataset = DisjointTimeBasedInitializerDataset(self.metadata.dataset_path, self.metadata.data_table_path, init_config)

            sampler = SequentialSampler(init_dataset)
            dataloader = DataLoader(init_dataset, num_workers=workers, collate_fn=self._collate_fn, worker_init_fn=DisjointTimeBasedInitializerDataset.worker_init_fn, persistent_workers=False, sampler=sampler)

            if workers == 0:
                init_dataset.pytables_worker_init()

            ts_ids_to_take = []

            for ts_id, data in enumerate(tqdm(dataloader, total=len(init_config.ts_row_ranges))):
                init_dataset_return: InitDatasetReturn = data[0]

                if init_dataset_return.is_under_nan_threshold:
                    ts_ids_to_take.append(ts_id)

                    # updates inner preprocessors passed from InitDataset
                    fitted_inner_index = 0
                    for inner_preprocess_order in group.preprocess_inner_orders:
                        if inner_preprocess_order.should_be_fitted:
                            if inner_preprocess_order.preprocess_type == PreprocessType.HANDLING_ANOMALIES:
                                self.dataset_config.anomaly_handlers[ts_id] = init_dataset_return.preprocess_fitted_instances[fitted_inner_index].instance
                            elif inner_preprocess_order.preprocess_type == PreprocessType.FILLING_GAPS:
                                self.dataset_config.train_fillers[ts_id] = init_dataset_return.preprocess_fitted_instances[fitted_inner_index].instance
                            else:
                                raise NotImplementedError()

                            fitted_inner_index += 1

                    # updates outer preprocessors based on passed train data from InitDataset
                    to_fit_outer_index = 0
                    for outer_preprocess_order in group.preprocess_outer_orders:
                        if outer_preprocess_order.should_be_fitted:
                            if outer_preprocess_order.preprocess_type == PreprocessType.TRANSFORMING:
                                self.dataset_config.transformers.partial_fit(init_dataset_return.train_data)
                            else:
                                raise NotImplementedError()

                            to_fit_outer_index += 1

            if workers == 0:
                init_dataset.cleanup()

            if len(ts_ids_to_take) == 0:
                raise ValueError("No valid time series left in train set after applying nan_threshold.")

            # Update config based on filtered time series
            if is_first_cycle:
                self.dataset_config.train_ts_row_ranges = self.dataset_config.train_ts_row_ranges[ts_ids_to_take]
                self.dataset_config.train_ts = self.dataset_config.train_ts[ts_ids_to_take]
                self.dataset_config.train_fillers = self.dataset_config.train_fillers[ts_ids_to_take]
                self.dataset_config.anomaly_handlers = self.dataset_config.anomaly_handlers[ts_ids_to_take]

                is_first_cycle = False

    def __initialize_config_for_non_fit_sets(self, init_config: DisjointTimeDatasetInitConfig, workers: int, set_name: str) -> np.ndarray:
        """Initializes config for provided time series without fitting. """
        init_dataset = DisjointTimeBasedInitializerDataset(self.metadata.dataset_path, self.metadata.data_table_path, init_config)

        sampler = SequentialSampler(init_dataset)
        dataloader = DataLoader(init_dataset, num_workers=workers, collate_fn=self._collate_fn, worker_init_fn=DisjointTimeBasedInitializerDataset.worker_init_fn, persistent_workers=False, sampler=sampler)

        if workers == 0:
            init_dataset.pytables_worker_init()

        ts_ids_to_take = []

        self.logger.info("Updating config for %s set.", set_name)
        for i, data in enumerate(tqdm(dataloader, total=len(init_config.ts_row_ranges))):
            init_dataset_return: InitDatasetReturn = data[0]

            if init_dataset_return.is_under_nan_threshold:
                ts_ids_to_take.append(i)

        if workers == 0:
            init_dataset.cleanup()

        if len(ts_ids_to_take) == 0:
            raise ValueError(f"No valid time series left in {set_name} set after applying nan_threshold.")

        return ts_ids_to_take

    def _update_export_config_copy(self) -> None:
        """
        Called at the end of [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize] or when changing config values. 

        Updates values of config used for saving config.
        """
        self._export_config_copy.database_name = self.metadata.database_name

        self._export_config_copy.train_ts = self.dataset_config.train_ts.copy() if self.dataset_config.has_train() else None
        self._export_config_copy.val_ts = self.dataset_config.val_ts.copy() if self.dataset_config.has_val() else None
        self._export_config_copy.test_ts = self.dataset_config.test_ts.copy() if self.dataset_config.has_test() else None

        self._export_config_copy.sliding_window_size = self.dataset_config.sliding_window_size
        self._export_config_copy.sliding_window_prediction_size = self.dataset_config.sliding_window_prediction_size
        self._export_config_copy.sliding_window_step = self.dataset_config.sliding_window_step
        self._export_config_copy.set_shared_size = self.dataset_config.set_shared_size

        super(DisjointTimeBasedCesnetDataset, self)._update_export_config_copy()

    def _get_singular_time_series_dataset(self, parent_dataset: DisjointTimeBasedSplittedDataset, ts_id: int) -> DisjointTimeBasedSplittedDataset:
        """Returns dataset for single time series """

        temp = np.where(np.isin(parent_dataset.load_config.ts_row_ranges[self.metadata.ts_id_name], [ts_id]))[0]

        if len(temp) == 0:
            raise ValueError(f"ts_id {ts_id} was not found in valid time series for this set. Available time series are: {parent_dataset.load_config.ts_row_ranges[self.metadata.ts_id_name]}")

        time_series_position = temp[0]

        split_load_config = parent_dataset.load_config.create_split_copy(slice(time_series_position, time_series_position + 1))

        dataset = DisjointTimeBasedSplittedDataset(self.metadata.dataset_path, self.metadata.data_table_path, split_load_config, 0)
        self.logger.debug("Singular time series dataset initiliazed.")

        return dataset

    def _get_data_for_plot(self, ts_id: int, feature_indices: np.ndarray[int], time_format: TimeFormat) -> tuple[np.ndarray, np.ndarray]:
        """Dataset type specific retrieval of data. """

        train_id_result, val_id_result, test_id_result = None, None, None

        if (self.dataset_config.has_train()):
            train_id_result = np.argwhere(np.isin(self.dataset_config.train_ts, ts_id)).ravel()
        if (self.dataset_config.has_val()):
            val_id_result = np.argwhere(np.isin(self.dataset_config.val_ts, ts_id)).ravel()
        if (self.dataset_config.has_test()):
            test_id_result = np.argwhere(np.isin(self.dataset_config.test_ts, ts_id)).ravel()

        data = None
        time_period = None

        if self.dataset_config.has_train() and len(train_id_result) > 0:
            data = self.__get_ts_data_for_plot(self.train_dataset, ts_id, feature_indices)
            time_period = self.get_data_about_set(SplitType.TRAIN)[time_format]
            self.logger.debug("Valid ts_id found: %d", train_id_result[0])

        elif self.dataset_config.has_val() and len(val_id_result) > 0:
            data = self.__get_ts_data_for_plot(self.val_dataset, ts_id, feature_indices)
            time_period = self.get_data_about_set(SplitType.VAL)[time_format]
            self.logger.debug("Valid ts_id found: %d", val_id_result[0])

        elif self.dataset_config.has_test() and len(test_id_result) > 0:
            data = self.__get_ts_data_for_plot(self.test_dataset, ts_id, feature_indices)
            time_period = self.get_data_about_set(SplitType.TEST)[time_format]
            self.logger.debug("Valid ts_id found: %d", test_id_result[0])
        else:
            raise ValueError(f"Invalid ts_id '{ts_id}'. The provided ts_id is not found in the available time series IDs.", self.dataset_config.train_ts, self.dataset_config.val_ts, self.dataset_config.test_ts)

        return data, time_period

    def __get_ts_data_for_plot(self, dataset: DisjointTimeBasedSplittedDataset, ts_id: int, feature_indices: list[int]):
        dataset = self._get_singular_time_series_dataset(dataset, ts_id)

        dataloader = self.dataloader_factory.create_dataloader(dataset, self.dataset_config, 0, True, None)

        temp_data = dataset_loaders.create_numpy_from_dataloader(dataloader, np.array([ts_id]), dataset.load_config.time_format, dataset.load_config.include_time, DatasetType.TIME_BASED, True)

        if (dataset.load_config.time_format == TimeFormat.DATETIME and dataset.load_config.include_time):
            temp_data = temp_data[0]

        temp_data = temp_data[0][:, feature_indices]

        return temp_data
