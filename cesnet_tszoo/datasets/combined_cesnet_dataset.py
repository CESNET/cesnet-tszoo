from datetime import datetime, timezone
from typing import Optional, Literal
from dataclasses import dataclass, field
from numbers import Number

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from cesnet_tszoo.utils.enums import SplitType, TimeFormat, DatasetType, TransformerType, FillerType
from cesnet_tszoo.configs.combined_config import CombinedConfig
from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.pytables_data.combined_initializer_dataset import CombinedInitializerDataset
from cesnet_tszoo.pytables_data.splitted_dataset import SplittedDataset
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, TIME_COLUMN_NAME


@dataclass
class CombinedCesnetDataset(CesnetDataset):

    dataset_config: Optional[CombinedConfig] = field(default=None, init=False)

    train_dataset: Optional[SplittedDataset] = field(default=None, init=False)
    val_dataset: Optional[SplittedDataset] = field(default=None, init=False)
    test_dataset: Optional[SplittedDataset] = field(default=None, init=False)
    all_dataset: Optional[SplittedDataset] = field(default=None, init=False)

    train_dataloader: Optional[DataLoader] = field(default=None, init=False)
    val_dataloader: Optional[DataLoader] = field(default=None, init=False)
    test_dataloader: Optional[DataLoader] = field(default=None, init=False)
    all_dataloader: Optional[DataLoader] = field(default=None, init=False)

    dataset_type: DatasetType = field(default=DatasetType.COMBINED, init=False)

    _export_config_copy: Optional[CombinedConfig] = field(default=None, init=False)

    def set_dataset_config_and_initialize(self, dataset_config: CombinedConfig, display_config_details: bool = True, workers: int | Literal["config"] = "config") -> None:
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

        assert isinstance(dataset_config, CombinedConfig), "CombinedCesnetDataset can only use CombinedConfig."

        super(CombinedCesnetDataset, self).set_dataset_config_and_initialize(dataset_config, display_config_details, workers)

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
                                             all_batch_size: int | Literal["config"] = "config",
                                             fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None | Literal["config"] = "config",
                                             transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None | Literal["config"] = "config",
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
            partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers. `Defaults: config`.    
            train_workers: Number of workers for loading training data. `Defaults: config`.
            val_workers: Number of workers for loading validation data. `Defaults: config`.
            test_workers: Number of workers for loading test data. `Defaults: config`.
            all_workers: Number of workers for loading all data.  `Defaults: config`.
            init_workers: Number of workers for dataset configuration. `Defaults: config`.                          
            workers: How many workers to use when updating configuration. `Defaults: config`.  
            display_config_details: Whether config details should be displayed after configuration. `Defaults: False`. 
        """

        return super(CombinedCesnetDataset, self).update_dataset_config_and_initialize(default_values, sliding_window_size, sliding_window_prediction_size, sliding_window_step, set_shared_size, train_batch_size, val_batch_size, test_batch_size, all_batch_size, fill_missing_with, transform_with, "config", partial_fit_initialized_transformers, train_workers, val_workers, test_workers, all_workers, init_workers, workers, display_config_details)

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
        elif about == SplitType.ALL:
            time_period = self.dataset_config.all_time_period
            time_series = self.dataset_config.all_ts

        datetime_temp = np.array([datetime.fromtimestamp(time, timezone.utc) for time in self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]]])

        result["ts_ids"] = time_series.copy()
        result[TimeFormat.ID_TIME] = time_period[ID_TIME_COLUMN_NAME].copy()
        result[TimeFormat.DATETIME] = datetime_temp.copy()
        result[TimeFormat.UNIX_TIME] = self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]].copy()
        result[TimeFormat.SHIFTED_UNIX_TIME] = self.time_indices[TIME_COLUMN_NAME][time_period[ID_TIME_COLUMN_NAME]] - self.time_indices[TIME_COLUMN_NAME][0]

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
            self.train_dataset = SplittedDataset(self.dataset_path,
                                                 self.dataset_config._get_table_data_path(),
                                                 self.dataset_config.ts_id_name,
                                                 self.dataset_config.train_ts_row_ranges,
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

        if self.dataset_config.has_val():
            self.val_dataset = SplittedDataset(self.dataset_path,
                                               self.dataset_config._get_table_data_path(),
                                               self.dataset_config.ts_id_name,
                                               self.dataset_config.val_ts_row_ranges,
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

        if self.dataset_config.has_test():
            self.test_dataset = SplittedDataset(self.dataset_path,
                                                self.dataset_config._get_table_data_path(),
                                                self.dataset_config.ts_id_name,
                                                self.dataset_config.test_ts_row_ranges,
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

    def _initialize_transformers_and_details(self, workers: int) -> None:
        """
        Called in [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]. 

        Goes through data to validate time series against `nan_threshold`, fit/partial fit `transformers` and prepare `fillers`.
        """

        all_ts_ids_to_take = np.array([])

        if self.dataset_config.has_train():
            updated_ts_row_ranges, updated_ts_ids, updated_fillers = self.__initialize_transformers_and_details_for_set(self.dataset_config.train_ts, self.dataset_config.train_ts_row_ranges, self.dataset_config.train_time_period,
                                                                                                                        self.dataset_config.train_fillers, workers, "train")
            self.dataset_config.train_ts = updated_ts_ids
            self.dataset_config.train_ts_row_ranges = updated_ts_row_ranges
            self.dataset_config.train_fillers = updated_fillers

            all_ts_ids_to_take = np.concatenate([all_ts_ids_to_take, updated_ts_ids]).astype(np.int32)

            self.logger.debug("Train set updated: %s time series left.", len(updated_ts_ids))

        if self.dataset_config.has_val():
            updated_ts_row_ranges, updated_ts_ids, updated_fillers = self.__initialize_transformers_and_details_for_set(self.dataset_config.val_ts, self.dataset_config.val_ts_row_ranges, self.dataset_config.val_time_period,
                                                                                                                        self.dataset_config.val_fillers, workers, "val")
            self.dataset_config.val_ts = updated_ts_ids
            self.dataset_config.val_ts_row_ranges = updated_ts_row_ranges
            self.dataset_config.val_fillers = updated_fillers

            all_ts_ids_to_take = np.concatenate([all_ts_ids_to_take, updated_ts_ids]).astype(np.int32)

            self.logger.debug("Val set updated: %s time series left.", len(updated_ts_ids))

        if self.dataset_config.has_test():
            updated_ts_row_ranges, updated_ts_ids, updated_fillers = self.__initialize_transformers_and_details_for_set(self.dataset_config.test_ts, self.dataset_config.test_ts_row_ranges, self.dataset_config.test_time_period,
                                                                                                                        self.dataset_config.test_fillers, workers, "test")
            self.dataset_config.test_ts = updated_ts_ids
            self.dataset_config.test_ts_row_ranges = updated_ts_row_ranges
            self.dataset_config.test_fillers = updated_fillers

            all_ts_ids_to_take = np.concatenate([all_ts_ids_to_take, updated_ts_ids]).astype(np.int32)

            self.logger.debug("Test set updated: %s time series left.", len(updated_ts_ids))

        _, idx = np.unique(all_ts_ids_to_take, True, False, False)
        idx = np.sort(idx)
        all_ts_ids_to_take = all_ts_ids_to_take[idx]
        mask = np.isin(self.dataset_config.all_ts, all_ts_ids_to_take)

        self.dataset_config.used_ts_row_ranges = self.dataset_config.all_ts_row_ranges[mask]
        self.dataset_config.used_ts_ids = self.dataset_config.all_ts[mask]
        self.dataset_config.used_times = self.dataset_config.all_time_period
        self.dataset_config.used_fillers = None if self.dataset_config.all_fillers is None else self.dataset_config.all_fillers[mask]

    def _update_export_config_copy(self) -> None:
        """
        Called at the end of [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize] or when changing config values. 

        Updates values of config used for saving config.
        """
        self._export_config_copy.database_name = self.database_name

        self._export_config_copy.train_ts = self.dataset_config.train_ts.copy() if self.dataset_config.has_train() else None
        self._export_config_copy.val_ts = self.dataset_config.val_ts.copy() if self.dataset_config.has_val() else None
        self._export_config_copy.test_ts = self.dataset_config.test_ts.copy() if self.dataset_config.has_test() else None

        self._export_config_copy.sliding_window_size = self.dataset_config.sliding_window_size
        self._export_config_copy.sliding_window_prediction_size = self.dataset_config.sliding_window_prediction_size
        self._export_config_copy.sliding_window_step = self.dataset_config.sliding_window_step
        self._export_config_copy.set_shared_size = self.dataset_config.set_shared_size

        super(CombinedCesnetDataset, self)._update_export_config_copy()

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

    def __initialize_transformers_and_details_for_set(self, ts_ids, ts_row_ranges, time_period, fillers, workers, set_name):
        init_dataset = CombinedInitializerDataset(self.dataset_path,
                                                  self.dataset_config._get_table_data_path(),
                                                  self.dataset_config.ts_id_name,
                                                  ts_row_ranges,
                                                  time_period,
                                                  self.dataset_config.features_to_take,
                                                  self.dataset_config.indices_of_features_to_take_no_ids,
                                                  self.dataset_config.default_values,
                                                  fillers)

        sampler = SequentialSampler(init_dataset)
        dataloader = DataLoader(init_dataset, num_workers=workers, collate_fn=self._collate_fn, worker_init_fn=CombinedInitializerDataset.worker_init_fn, persistent_workers=False, sampler=sampler)

        if workers == 0:
            init_dataset.pytables_worker_init()

        ts_ids_to_take = []

        self.logger.info("Updating config for %s set.", set_name)
        for i, data in enumerate(tqdm(dataloader, total=len(ts_ids))):
            data, count_values = data[0]

            # Filter time series based on missing data threshold
            missing_train_percentage = count_values[1] / (count_values[0] + count_values[1])

            if missing_train_percentage <= self.dataset_config.nan_threshold:
                ts_ids_to_take.append(i)

                # Fit transformers if required
                if self.dataset_config.transform_with is not None and data is not None and (not self.dataset_config.are_transformers_premade or self.dataset_config.partial_fit_initialized_transformers):

                    if self.dataset_config.are_transformers_premade and self.dataset_config.partial_fit_initialized_transformers:
                        self.dataset_config.transformers.partial_fit(data)
                    else:
                        self.dataset_config.transformers.partial_fit(data)

        if workers == 0:
            init_dataset.cleanup()

        if len(ts_ids_to_take) == 0:
            raise ValueError(f"No valid time series left in {set_name} set after applying nan_threshold.")

        # Update config based on filtered time series
        updated_ts_row_ranges = ts_row_ranges[ts_ids_to_take]
        updated_ts_ids = ts_ids[ts_ids_to_take]
        updated_fillers = None if self.dataset_config.fill_missing_with is None else fillers[ts_ids_to_take]

        return updated_ts_row_ranges, updated_ts_ids, updated_fillers
