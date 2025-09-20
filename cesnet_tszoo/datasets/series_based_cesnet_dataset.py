from datetime import datetime, timezone
from typing import Optional, Literal
from dataclasses import dataclass, field
from numbers import Number

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from cesnet_tszoo.utils.enums import SplitType, TimeFormat, DataloaderOrder, TransformerType, FillerType, DatasetType, AnomalyHandlerType
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME
from cesnet_tszoo.configs.series_based_config import SeriesBasedConfig
from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.pytables_data.series_based_dataset import SeriesBasedDataset
from cesnet_tszoo.pytables_data.series_based_initializer_dataset import SeriesBasedInitializerDataset
from cesnet_tszoo.datasets.loaders import create_numpy_from_dataloader
from cesnet_tszoo.utils.transformer import Transformer


@dataclass
class SeriesBasedCesnetDataset(CesnetDataset):
    """
    This class is used for series-based returning of data. Can be created by using [`get_dataset`][cesnet_tszoo.datasets.cesnet_database.CesnetDatabase.get_dataset] with parameter `dataset_type` = `DatasetType.SERIES_BASED`.

    Series-based means batch size affects number of returned time series in one batch. Which times for each time series are returned does not change.

    The dataset provides multiple ways to access the data:

    - **Iterable PyTorch DataLoader**: For batch processing.
    - **Pandas DataFrame**: For loading the entire training, validation, test or all set at once.
    - **Numpy array**: For loading the entire training, validation, test or all set at once.      
    - See [loading data][loading-data] for more details.

    The dataset is stored in a [PyTables](https://www.pytables.org/) database. The internal `SeriesBasedDataset` and `SeriesBasedInitializerDataset` classes (used only when calling [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize]) act as wrappers that implement the PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 
    interface. These wrappers are compatible with PyTorch’s [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), providing efficient parallel data loading. 

    The dataset configuration is done through the [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig] class.     

    **Intended usage:**

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.cesnet_database.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_numpy].     

    Alternatively you can use [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark]

    1. Call [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark] with the desired benchmark. You can use your own saved benchmark or you can use already built-in one. This will download the dataset and annotations (if available) if they have not been previously downloaded.
    2. Retrieve the initialized dataset using [`get_initialized_dataset`][cesnet_tszoo.benchmarks.Benchmark.get_initialized_dataset]. This will provide a dataset that is ready to use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_numpy].       

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
        additional_data: Available small datasets. Can get them by calling [`get_additional_data`][cesnet_tszoo.datasets.cesnet_dataset.CesnetDataset.get_additional_data] with their name.

    Attributes:
        time_indices: Available time IDs for the dataset.
        ts_indices: Available time series IDs for the dataset.
        annotations: Annotations for the selected dataset.
        logger: Logger for displaying information.  
        imported_annotations_ts_identifier: Identifier for the imported annotations of type `AnnotationType.TS_ID`.
        imported_annotations_time_identifier: Identifier for the imported annotations of type `AnnotationType.ID_TIME`.
        imported_annotations_both_identifier: Identifier for the imported annotations of type `AnnotationType.BOTH`.   

    The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize] is called.

    Attributes:
        dataset_type: Type of this dataset.
        dataset_config: Configuration of the dataset.
        train_dataset: Training set as a `SeriesBasedDataset` instance wrapping the PyTables database.
        val_dataset: Validation set as a `SeriesBasedDataset` instance wrapping the PyTables database.
        test_dataset: Test set as a `SeriesBasedDataset` instance wrapping the PyTables database.
        all_dataset: All set as a `SeriesBasedDataset` instance wrapping the PyTables database.        
        train_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training set.
        val_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation set.
        test_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for test set.
        all_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for all set.          
    """

    dataset_config: Optional[SeriesBasedConfig] = field(default=None, init=False)

    train_dataset: Optional[SeriesBasedDataset] = field(default=None, init=False)
    val_dataset: Optional[SeriesBasedDataset] = field(default=None, init=False)
    test_dataset: Optional[SeriesBasedDataset] = field(default=None, init=False)
    all_dataset: Optional[SeriesBasedDataset] = field(default=None, init=False)

    dataset_type: DatasetType = field(default=DatasetType.SERIES_BASED, init=False)

    _export_config_copy: Optional[SeriesBasedConfig] = field(default=None, init=False)

    def set_dataset_config_and_initialize(self, dataset_config: SeriesBasedConfig, display_config_details: bool = True, workers: int | Literal["config"] = "config") -> None:
        """
        Initialize training set, validation est, test set etc.. This method must be called before any data can be accessed. It is required for the final initialization of [`dataset_config`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig].

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
        assert isinstance(dataset_config, SeriesBasedConfig), f"This config is used for dataset of type '{dataset_config.dataset_type}'. Meanwhile this dataset is of type '{self.dataset_type}'."

        super(SeriesBasedCesnetDataset, self).set_dataset_config_and_initialize(dataset_config, display_config_details, workers)

    def update_dataset_config_and_initialize(self,
                                             default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None | Literal["config"] = "config",
                                             train_batch_size: int | Literal["config"] = "config",
                                             val_batch_size: int | Literal["config"] = "config",
                                             test_batch_size: int | Literal["config"] = "config",
                                             all_batch_size: int | Literal["config"] = "config",
                                             fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None | Literal["config"] = "config",
                                             transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None | Literal["config"] = "config",
                                             handle_anomalies_with: type | AnomalyHandlerType | Literal["z-score", "interquartile_range"] | None | Literal["config"] = "config",
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
        | `train_batch_size`                      | Number of samples per batch for train set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details. |
        | `val_batch_size`                        | Number of samples per batch for val set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.   |
        | `test_batch_size`                       | Number of samples per batch for test set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.  |
        | `all_batch_size`                        | Number of samples per batch for all set. Affected by whether the dataset is series-based or time-based. Refer to relevant config for details.   |                   
        | `fill_missing_with`                     | Defines how to fill missing values in the dataset.                                                                                              |     
        | `transform_with`                        | Defines the transformer to transform the dataset.                                                                                               | 
        | `handle_anomalies_with`                 | Defines the anomaly handler to handle anomalies in the train set.                                                                               |     
        | `partial_fit_initialized_transformers`  | If `True`, partial fitting on train set is performed when using initiliazed transformers.                                                       |   
        | `train_workers`                         | Number of workers for loading training data.                                                                                                    |
        | `val_workers`                           | Number of workers for loading validation data.                                                                                                  |
        | `test_workers`                          | Number of workers for loading test data.                                                                                                        |
        | `all_workers`                           | Number of workers for loading all data.                                                                                                         |     
        | `init_workers`                          | Number of workers for dataset configuration.                                                                                                    |                        

        Parameters:
            default_values: Default values for missing data, applied before fillers. `Defaults: config`.            
            train_batch_size: Number of samples per batch for train set. `Defaults: config`.
            val_batch_size: Number of samples per batch for val set. `Defaults: config`.
            test_batch_size: Number of samples per batch for test set. `Defaults: config`.
            all_batch_size: Number of samples per batch for all set. `Defaults: config`.                    
            fill_missing_with: Defines how to fill missing values in the dataset. `Defaults: config`. 
            transform_with: Defines the transformer to transform the dataset. `Defaults: config`.  
            handle_anomalies_with: Defines the anomaly handler to handle anomalies in the train set. `Defaults: config`.
            partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers. `Defaults: config`.    
            train_workers: Number of workers for loading training data. `Defaults: config`.
            val_workers: Number of workers for loading validation data. `Defaults: config`.
            test_workers: Number of workers for loading test data. `Defaults: config`.
            all_workers: Number of workers for loading all data.  `Defaults: config`.
            init_workers: Number of workers for dataset configuration. `Defaults: config`.                          
            workers: How many workers to use when updating configuration. `Defaults: config`.  
            display_config_details: Whether config details should be displayed after configuration. `Defaults: False`. 
        """

        return super(SeriesBasedCesnetDataset, self).update_dataset_config_and_initialize(default_values, "config", "config", "config", "config", train_batch_size, val_batch_size, test_batch_size, all_batch_size, fill_missing_with, transform_with, handle_anomalies_with, "config", partial_fit_initialized_transformers, train_workers, val_workers, test_workers, all_workers, init_workers, workers, display_config_details)

    def get_data_about_set(self, about: SplitType | Literal["train", "val", "test", "all"]) -> dict:
        """
        Retrieves data related to the specified set.

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

        time_period = self.dataset_config.time_period

        result = {}

        if about == SplitType.TRAIN:
            if not self.dataset_config.has_train():
                raise ValueError("Train set is not used.")
            ts_ids = self.dataset_config.train_ts
        elif about == SplitType.VAL:
            if not self.dataset_config.has_val():
                raise ValueError("Val set is not used.")
            ts_ids = self.dataset_config.val_ts
        elif about == SplitType.TEST:
            if not self.dataset_config.has_test():
                raise ValueError("Test set is not used.")
            ts_ids = self.dataset_config.test_ts
        elif about == SplitType.ALL:
            if not self.dataset_config.has_all():
                raise ValueError("All set is not used.")
            ts_ids = self.dataset_config.all_ts
        else:
            raise NotImplementedError("Should not happen")

        datetime_temp = np.array([datetime.fromtimestamp(time, tz=timezone.utc) for time in self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]]])

        result["ts_ids"] = ts_ids.copy()
        result[TimeFormat.ID_TIME] = time_period[ID_TIME_COLUMN_NAME].copy()
        result[TimeFormat.DATETIME] = datetime_temp.copy()
        result[TimeFormat.UNIX_TIME] = self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]].copy()
        result[TimeFormat.SHIFTED_UNIX_TIME] = self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]] - self.time_indices[TIME_COLUMN_NAME][0]

        return result

    def _initialize_datasets(self) -> None:
        """Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize], this method initializes the set datasets (train, validation, test and all). """

        if self.dataset_config.has_train():
            self.train_dataset = SeriesBasedDataset(self.dataset_path,
                                                    self.dataset_config._get_table_data_path(),
                                                    self.dataset_config.ts_id_name,
                                                    self.dataset_config.train_ts_row_ranges,
                                                    self.dataset_config.time_period,
                                                    self.dataset_config.features_to_take,
                                                    self.dataset_config.indices_of_features_to_take_no_ids,
                                                    self.dataset_config.default_values,
                                                    self.dataset_config.train_fillers,
                                                    self.dataset_config.include_time,
                                                    self.dataset_config.include_ts_id,
                                                    self.dataset_config.time_format,
                                                    self.dataset_config.transformers,
                                                    self.dataset_config.anomaly_handlers)
            self.logger.debug("train_dataset initiliazed.")

        if self.dataset_config.has_val():
            self.val_dataset = SeriesBasedDataset(self.dataset_path,
                                                  self.dataset_config._get_table_data_path(),
                                                  self.dataset_config.ts_id_name,
                                                  self.dataset_config.val_ts_row_ranges,
                                                  self.dataset_config.time_period,
                                                  self.dataset_config.features_to_take,
                                                  self.dataset_config.indices_of_features_to_take_no_ids,
                                                  self.dataset_config.default_values,
                                                  self.dataset_config.val_fillers,
                                                  self.dataset_config.include_time,
                                                  self.dataset_config.include_ts_id,
                                                  self.dataset_config.time_format,
                                                  self.dataset_config.transformers,
                                                  None)
            self.logger.debug("val_dataset initiliazed.")

        if self.dataset_config.has_test():
            self.test_dataset = SeriesBasedDataset(self.dataset_path,
                                                   self.dataset_config._get_table_data_path(),
                                                   self.dataset_config.ts_id_name,
                                                   self.dataset_config.test_ts_row_ranges,
                                                   self.dataset_config.time_period,
                                                   self.dataset_config.features_to_take,
                                                   self.dataset_config.indices_of_features_to_take_no_ids,
                                                   self.dataset_config.default_values,
                                                   self.dataset_config.test_fillers,
                                                   self.dataset_config.include_time,
                                                   self.dataset_config.include_ts_id,
                                                   self.dataset_config.time_format,
                                                   self.dataset_config.transformers,
                                                   None)
            self.logger.debug("test_dataset initiliazed.")

        if self.dataset_config.has_all():
            self.all_dataset = SeriesBasedDataset(self.dataset_path,
                                                  self.dataset_config._get_table_data_path(),
                                                  self.dataset_config.ts_id_name,
                                                  self.dataset_config.all_ts_row_ranges,
                                                  self.dataset_config.time_period,
                                                  self.dataset_config.features_to_take,
                                                  self.dataset_config.indices_of_features_to_take_no_ids,
                                                  self.dataset_config.default_values,
                                                  self.dataset_config.all_fillers,
                                                  self.dataset_config.include_time,
                                                  self.dataset_config.include_ts_id,
                                                  self.dataset_config.time_format,
                                                  self.dataset_config.transformers,
                                                  None)
            self.logger.debug("all_dataset initiliazed.")

    def _initialize_transformers_and_details(self, workers: int) -> None:
        """
        Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize]. 

        Goes through data to validate time series against `nan_threshold`, partial fit `transformers`, fit `anomaly handlers` and prepare `fillers`.
        """

        init_dataset = SeriesBasedInitializerDataset(self.dataset_path,
                                                     self.dataset_config._get_table_data_path(),
                                                     self.dataset_config.ts_id_name,
                                                     self.dataset_config.train_ts_row_ranges,
                                                     self.dataset_config.val_ts_row_ranges,
                                                     self.dataset_config.test_ts_row_ranges,
                                                     self.dataset_config.all_ts_row_ranges,
                                                     self.dataset_config.time_period,
                                                     self.dataset_config.features_to_take,
                                                     self.dataset_config.indices_of_features_to_take_no_ids,
                                                     self.dataset_config.default_values,
                                                     self.dataset_config.all_fillers,
                                                     self.dataset_config.anomaly_handlers)

        sampler = SequentialSampler(init_dataset)
        dataloader = DataLoader(init_dataset,
                                num_workers=workers,
                                collate_fn=self._collate_fn,
                                worker_init_fn=SeriesBasedInitializerDataset.worker_init_fn,
                                persistent_workers=False,
                                sampler=sampler)

        if workers == 0:
            init_dataset.pytables_worker_init()

        train_ts_ids_to_take = []
        val_ts_ids_to_take = []
        test_ts_ids_to_take = []
        all_ts_ids_to_take = []

        self.logger.info("Updating config on train/val/test/all and selected time period.")
        for i, data in enumerate(tqdm(dataloader)):
            train_data, count_values, is_train, is_val, is_test, offsetted_idx, anomaly_handler = data[0]

            missing_percentage = count_values[1] / (count_values[0] + count_values[1])

            # Filter time series based on missing data threshold
            if missing_percentage <= self.dataset_config.nan_threshold:
                if is_train:
                    self.dataset_config.anomaly_handlers[i] = anomaly_handler
                    train_ts_ids_to_take.append(offsetted_idx)
                elif is_val:
                    val_ts_ids_to_take.append(offsetted_idx)
                elif is_test:
                    test_ts_ids_to_take.append(offsetted_idx)

                all_ts_ids_to_take.append(i)

                # Partial fit transformer on train data if applicable
                if is_train and (not self.dataset_config.transformer_factory.has_already_initialized or self.dataset_config.partial_fit_initialized_transformers):
                    self.dataset_config.transformers.partial_fit(train_data)

        if workers == 0:
            init_dataset.cleanup()

        # Update sets based on filtered time series
        if self.dataset_config.has_train():
            if len(train_ts_ids_to_take) == 0:
                raise ValueError("No time series left in training set after applying nan_threshold.")
            self.dataset_config.train_ts_row_ranges = self.dataset_config.train_ts_row_ranges[train_ts_ids_to_take]
            self.dataset_config.train_ts = self.dataset_config.train_ts[train_ts_ids_to_take]
            self.dataset_config.train_fillers = self.dataset_config.train_fillers[train_ts_ids_to_take]
            self.dataset_config.anomaly_handlers = self.dataset_config.anomaly_handlers[train_ts_ids_to_take]

            self.logger.debug("Train set updated: %s time series left.", len(train_ts_ids_to_take))

        if self.dataset_config.has_val():
            if len(val_ts_ids_to_take) == 0:
                raise ValueError("No time series left in validation set after applying nan_threshold.")
            self.dataset_config.val_ts_row_ranges = self.dataset_config.val_ts_row_ranges[val_ts_ids_to_take]
            self.dataset_config.val_ts = self.dataset_config.val_ts[val_ts_ids_to_take]
            self.dataset_config.val_fillers = self.dataset_config.val_fillers[val_ts_ids_to_take]

            self.logger.debug("Validation set updated: %s time series selected.", len(val_ts_ids_to_take))

        if self.dataset_config.has_test():
            if len(test_ts_ids_to_take) == 0:
                raise ValueError("No time series left in test set after applying nan_threshold.")
            self.dataset_config.test_ts_row_ranges = self.dataset_config.test_ts_row_ranges[test_ts_ids_to_take]
            self.dataset_config.test_ts = self.dataset_config.test_ts[test_ts_ids_to_take]
            self.dataset_config.test_fillers = self.dataset_config.test_fillers[test_ts_ids_to_take]

            self.logger.debug("Test set updated: %s time series selected.", len(test_ts_ids_to_take))

        if self.dataset_config.has_all():
            if len(all_ts_ids_to_take) == 0:
                raise ValueError("No series left in all set after applying nan_threshold.")
            self.dataset_config.all_ts = self.dataset_config.all_ts[all_ts_ids_to_take]
            self.dataset_config.all_ts_row_ranges = self.dataset_config.all_ts_row_ranges[all_ts_ids_to_take]
            self.dataset_config.all_fillers = self.dataset_config.all_fillers[all_ts_ids_to_take]

            self.logger.debug("All set updated: %s time series selected.", len(all_ts_ids_to_take))

        self.logger.info("Dataset initialization complete. Configuration updated.")

    def _update_export_config_copy(self) -> None:
        """
        Called at the end of [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize]. 

        Updates values of config used for saving config.
        """

        self._export_config_copy.database_name = self.database_name

        if self.dataset_config.has_train():
            self._export_config_copy.train_ts = self.dataset_config.train_ts.copy()
            self.logger.debug("Updated train_ts of _export_config_copy.")

        if self.dataset_config.has_val():
            self._export_config_copy.val_ts = self.dataset_config.val_ts.copy()
            self.logger.debug("Updated val_ts of _export_config_copy.")

        if self.dataset_config.has_test():
            self._export_config_copy.test_ts = self.dataset_config.test_ts.copy()
            self.logger.debug("Updated test_ts of _export_config_copy.")

        super(SeriesBasedCesnetDataset, self)._update_export_config_copy()

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

    def _get_singular_time_series_dataset(self, parent_dataset: SeriesBasedDataset, ts_id: int) -> SeriesBasedDataset:
        """Returns dataset for single time series """

        temp = np.where(np.isin(parent_dataset.ts_row_ranges[self.ts_id_name], [ts_id]))[0]

        if len(temp) == 0:
            raise ValueError(f"ts_id {ts_id} was not found in valid time series for this set. Available time series are: {parent_dataset.ts_row_ranges[self.ts_id_name]}")

        time_series_position = temp[0]

        filler = parent_dataset.fillers[time_series_position:time_series_position + 1]
        transformer = parent_dataset.transformers
        anomaly_handler = None if parent_dataset.anomaly_handlers is None else parent_dataset.anomaly_handlers[time_series_position:time_series_position + 1]

        dataset = SeriesBasedDataset(self.dataset_path,
                                     self.dataset_config._get_table_data_path(),
                                     self.dataset_config.ts_id_name,
                                     parent_dataset.ts_row_ranges[time_series_position: time_series_position + 1],
                                     parent_dataset.time_period,
                                     self.dataset_config.features_to_take,
                                     self.dataset_config.indices_of_features_to_take_no_ids,
                                     self.dataset_config.default_values,
                                     filler,
                                     self.dataset_config.include_time,
                                     self.dataset_config.include_ts_id,
                                     self.dataset_config.time_format,
                                     transformer,
                                     anomaly_handler
                                     )
        self.logger.debug("Singular time series dataset initiliazed.")

        return dataset

    def _get_dataloader(self, dataset: SeriesBasedDataset, workers: int | Literal["config"], take_all: bool, batch_size: int, **kwargs) -> DataLoader:
        """Set series based dataloader for this dataset. """

        default_kwargs = {'order': DataloaderOrder.SEQUENTIAL}
        kwargs = {**default_kwargs, **kwargs}

        return self._get_series_based_dataloader(dataset, workers, take_all, batch_size, kwargs["order"])

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

    def __get_ts_data_for_plot(self, dataset: SeriesBasedDataset, ts_id: int, feature_indices: list[int]):
        dataset = self._get_singular_time_series_dataset(dataset, ts_id)
        dataloader = self._get_series_based_dataloader(dataset, 0, True, None)

        temp_data = create_numpy_from_dataloader(dataloader, np.array([ts_id]), dataset.time_format, dataset.include_time, DatasetType.SERIES_BASED, True)

        if (dataset.time_format == TimeFormat.DATETIME and dataset.include_time):
            temp_data = temp_data[0]

        temp_data = temp_data[0][:, feature_indices]

        return temp_data
