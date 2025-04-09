from typing import Literal
from datetime import datetime
from numbers import Number

import numpy as np
import numpy.typing as npt

from cesnet_tszoo.utils.filler import filler_from_input_to_type
from cesnet_tszoo.utils.scaler import scaler_from_input_to_scaler_type, Scaler
from cesnet_tszoo.utils.utils import get_abbreviated_list_string
from cesnet_tszoo.utils.enums import FillerType, ScalerType, TimeFormat, DataloaderOrder
from cesnet_tszoo.configs.base_config import DatasetConfig


class SeriesBasedConfig(DatasetConfig):
    """
    This class is used for configuring the [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset].

    Used to configure the following:

    - Train, validation, test, all sets (time period, sizes, features)
    - Handling missing values (default values, [`fillers`][cesnet_tszoo.utils.filler])
    - Data transformation using [`scalers`][cesnet_tszoo.utils.scaler]
    - Dataloader options (train/val/test/all/init workers, batch size, train loading order)
    - Plotting

    **Important Notes:**

    - Custom fillers must inherit from the [`fillers`][cesnet_tszoo.utils.filler.Filler] base class.
    - Fillers can carry over values from the train set to the validation and test sets. For example, [`ForwardFiller`][cesnet_tszoo.utils.filler.ForwardFiller] can carry over values from previous sets.
    - It is recommended to use the [`scalers`][cesnet_tszoo.utils.scaler.Scaler] base class, though this is not mandatory as long as it meets the required methods.
        - If a scaler is already initialized and `partial_fit_initialized_scalers` is `False`, the scaler does not require `partial_fit`.
        - Otherwise, the scaler must support `partial_fit`.
        - Scalers must implement `transform` method.
        - Both `partial_fit` and `transform` methods must accept an input of type `np.ndarray` with shape `(times, features)`.
    - `train_ts`, `val_ts`, and `test_ts` must not contain any overlapping time series IDs.

    For available configuration options, refer to [here][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig--configuration-options].

    Attributes:
        used_train_workers: Tracks the number of train workers in use. Helps determine if the train dataloader should be recreated based on worker changes.
        used_val_workers: Tracks the number of validation workers in use. Helps determine if the validation dataloader should be recreated based on worker changes.
        used_test_workers: Tracks the number of test workers in use. Helps determine if the test dataloader should be recreated based on worker changes.
        used_all_workers: Tracks the total number of all workers in use. Helps determine if the all dataloader should be recreated based on worker changes.
        import_identifier: Tracks the name of the config upon import. None if not imported.
        logger: Logger for displaying information.   

    The following attributes are initialized when [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize] is called:

    Attributes:
        all_ts: If no specific sets (train/val/test) are provided, all time series IDs are used. When any set is defined, only the time series IDs in defined sets are used.
        train_ts_row_ranges: Initialized when `train_ts_id` is set. Contains time series IDs in train set with their respective time ID ranges.
        val_ts_row_ranges: Initialized when `val_ts_id` is set. Contains time series IDs in validation set with their respective time ID ranges.
        test_ts_row_ranges: Initialized when `test_ts` is set. Contains time series IDs in test set with their respective time ID ranges.
        all_ts_row_ranges: Initialized when `all_ts` is set. Contains time series IDs in all set with their respective time ID ranges.
        display_time_period: Used to display the configured value of `time_period`.

        aggregation: The aggregation period used for the data.
        source_type: The source type of the data.
        database_name: Specifies which database this config applies to.
        scale_with_display: Used to display the configured type of `scale_with`.
        fill_missing_with_display: Used to display the configured type of `fill_missing_with`.
        features_to_take_without_ids: Features to be returned, excluding time or time series IDs.
        indices_of_features_to_take_no_ids: Indices of non-ID features in `features_to_take`.
        is_scaler_custom: Flag indicating whether the scaler is custom.
        is_filler_custom: Flag indicating whether the filler is custom.
        ts_id_name: Name of the time series ID, dependent on `source_type`.
        used_times: List of all times used in the configuration.
        used_ts_ids: List of all time series IDs used in the configuration.
        used_ts_row_ranges: List of time series IDs with their respective time ID ranges.
        used_fillers: List of all fillers used in the configuration.
        used_singular_train_time_series: Currently used singular train set time series for dataloader.
        used_singular_val_time_series: Currently used singular validation set time series for dataloader.
        used_singular_test_time_series: Currently used singular test set time series for dataloader.
        used_singular_all_time_series: Currently used singular all set time series for dataloader.             
        scalers: Prepared scalers for fitting/transforming. Can be one scaler, array of scalers or `None`.
        are_scalers_premade: Indicates whether the scalers are premade.
        has_train: Flag indicating whether the training set is in use.
        has_val: Flag indicating whether the validation set is in use.
        has_test: Flag indicating whether the test set is in use.
        has_all: Flag indicating whether the all set is in use.
        train_fillers: Fillers used in the train set. `None` if no filler is used or train set is not used.
        val_fillers: Fillers used in the validation set. `None` if no filler is used or validation set is not used.
        test_fillers: Fillers used in the test set. `None` if no filler is used or test set is not used.
        all_fillers: Fillers used for the all set. `None` if no filler is used or all set is not used.
        is_initialized: Flag indicating if the configuration has already been initialized. If true, config initialization will be skipped.          

    # Configuration options

    Attributes:
        time_period: Defines the time period for returning data from `train/val/test/all`. Can be a range of time IDs, a tuple of datetime objects or a float. Float value is equivalent to percentage of available times from start.
        train_ts: Defines which time series IDs are used in the training set. Can be a list of IDs, or an integer/float to specify a random selection. An `int` specifies the number of random time series, and a `float` specifies the proportion of available time series. 
                  `int` and `float` must be greater than 0, and a float should be smaller or equal to 1.0. Using `int` or `float` guarantees that no time series from other sets will be used. `Default: None`
        val_ts: Defines which time series IDs are used in the validation set. Same as `train_ts` but for the validation set. `Default: None`
        test_ts: Defines which time series IDs are used in the test set. Same as `train_ts` but for the test set. `Default: None`           
        features_to_take: Defines which features are used. `Default: "all"`
        default_values: Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature. `Default: "default"`
        train_batch_size: Batch size for the train dataloader. Affects number of returned time series in one batch. `Default: 32`
        val_batch_size: Batch size for the validation dataloader. Affects number of returned time series in one batch. `Default: 64`
        test_batch_size: Batch size for the test dataloader. Affects number of returned time series in one batch. `Default: 128`
        all_batch_size: Batch size for the all dataloader. Affects number of returned time series in one batch. `Default: 128`         
        fill_missing_with: Defines how to fill missing values in the dataset. Can pass enum [`FillerType`][cesnet_tszoo.utils.enums.FillerType] for built-in filler or pass a type of custom filler that must derive from [`Filler`][cesnet_tszoo.utils.filler.Filler] base class. `Default: None`
        scale_with: Defines the scaler used to transform the dataset. Can pass enum [`ScalerType`][cesnet_tszoo.utils.enums.ScalerType] for built-in scaler, pass a type of custom scaler or instance of already fitted scaler. `Default: None`
        partial_fit_initialized_scaler: If `True`, partial fitting on train set is performed when using initiliazed scaler. `Default: False`
        include_time: If `True`, time data is included in the returned values. `Default: True`
        include_ts_id: If `True`, time series IDs are included in the returned values. `Default: True`
        time_format: Format for the returned time data. When using TimeFormat.DATETIME, time will be returned as separate list along rest of the values. `Default: TimeFormat.ID_TIME`
        train_workers: Number of workers for loading training data. `0` means that the data will be loaded in the main process. `Default: 4`
        val_workers: Number of workers for loading validation data. `0` means that the data will be loaded in the main process. `Default: 3`
        test_workers: Number of workers for loading test data. `0` means that the data will be loaded in the main process. `Default: 2`
        all_workers: Number of workers for loading all data. `0` means that the data will be loaded in the main process. `Default: 4`
        init_workers: Number of workers for initial dataset processing during configuration. `0` means that the data will be loaded in the main process. `Default: 4`
        nan_threshold: Maximum allowable percentage of missing data. Time series exceeding this threshold are excluded. Time series over the threshold will not be used. Used for `train/val/test/all` separately. `Default: 1.0`
        train_dataloader_order: Defines the order of data returned by the training dataloader. `Default: DataloaderOrder.SEQUENTIAL`
        random_state: Fixes randomness for reproducibility during configuration and dataset initialization. `Default: None`                  
    """

    def __init__(self,
                 time_period: tuple[datetime, datetime] | range | float | Literal["all"],
                 train_ts: list[int] | npt.NDArray[np.int_] | float | int | None = None,
                 val_ts: list[int] | npt.NDArray[np.int_] | float | int | None = None,
                 test_ts: list[int] | npt.NDArray[np.int_] | float | int | None = None,
                 features_to_take: list[str] | Literal["all"] = "all",
                 default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None = "default",
                 train_batch_size: int = 32,
                 val_batch_size: int = 64,
                 test_batch_size: int = 128,
                 all_batch_size: int = 128,
                 fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None = None,
                 scale_with: type | ScalerType | Scaler | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_scaler", "l2_normalizer"] | None = None,
                 partial_fit_initialized_scaler: bool = False,
                 include_time: bool = True,
                 include_ts_id: bool = True,
                 time_format: TimeFormat | Literal["id_time", "datetime", "unix_time", "shifted_unix_time"] = TimeFormat.ID_TIME,
                 train_workers: int = 4,
                 val_workers: int = 3,
                 test_workers: int = 2,
                 all_workers: int = 4,
                 init_workers: int = 4,
                 nan_threshold: float = 1.0,
                 train_dataloader_order: DataloaderOrder | Literal["random", "sequential"] = DataloaderOrder.SEQUENTIAL,
                 random_state: int | None = None):

        self.time_period = time_period
        self.train_ts = train_ts
        self.val_ts = val_ts
        self.test_ts = test_ts

        self.all_ts = None
        self.train_ts_row_ranges = None
        self.val_ts_row_ranges = None
        self.test_ts_row_ranges = None
        self.all_ts_row_ranges = None
        self.display_time_period = None

        super(SeriesBasedConfig, self).__init__(features_to_take, default_values, None, None, 1, 0, train_batch_size, val_batch_size, test_batch_size, all_batch_size, fill_missing_with, scale_with, partial_fit_initialized_scaler, include_time, include_ts_id, time_format,
                                                train_workers, val_workers, test_workers, all_workers, init_workers, nan_threshold, False, True, train_dataloader_order, random_state)

    def _validate_construction(self) -> None:
        """Performs basic parameter validation to ensure correct configuration. More comprehensive validation, which requires dataset-specific data, is handled in [`_dataset_init`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig._dataset_init]. """

        super(SeriesBasedConfig, self)._validate_construction()

        if isinstance(self.time_period, (float, int)):
            self.time_period = float(self.time_period)
            assert self.time_period > 0.0, "time_period must be greater than 0"
            assert self.time_period <= 1.0, "time_period must be lower or equal to 1.0"

        split_float_total = 0

        if isinstance(self.train_ts, (float, int)):
            assert self.train_ts > 0, "train_ts must be greater than 0."
            if isinstance(self.train_ts, float):
                split_float_total += self.train_ts

        if isinstance(self.val_ts, (float, int)):
            assert self.val_ts > 0, "val_ts must be greater than 0"
            if isinstance(self.val_ts, float):
                split_float_total += self.val_ts

        if isinstance(self.test_ts, (float, int)):
            assert self.test_ts > 0, "test_ts must be greater than 0"
            if isinstance(self.test_ts, float):
                split_float_total += self.test_ts

        # Check if the total of float splits exceeds 1.0
        if split_float_total > 1.0:
            self.logger.error("The total of the float split sizes is greater than 1.0. Current total: %s", split_float_total)
            raise ValueError("Total value of used float split sizes can't be larger than 1.0.")

        self.logger.debug("Series-based configuration validated successfully.")

    def _get_train(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the training set. """
        return self.train_ts, self.time_period

    def _get_val(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the validation set. """
        return self.val_ts, self.time_period

    def _get_test(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the test set. """
        return self.test_ts, self.time_period

    def _get_all(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the all set. """
        return self.all_ts, self.time_period

    def _set_time_period(self, all_time_ids: np.ndarray) -> None:
        """Validates and filters the input time period based on the dataset and aggregation. """

        if self.time_period == "all":
            self.time_period = range(len(all_time_ids))
            self.logger.debug("Time period set to 'all'. Using all available time IDs, range: %s", self.time_period)
        elif isinstance(self.time_period, float):
            self.time_period = range(int(self.time_period * len(all_time_ids)))
            self.logger.debug("Time period set with float value. Using range: %s", self.time_period)

        self.time_period, self.display_time_period = self._process_time_period(self.time_period, all_time_ids)
        self.logger.debug("Processed time_period: %s, display_time_period: %s", self.time_period, self.display_time_period)

    def _set_ts(self, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray) -> None:
        """Validates and filters the input time series IDs based on the `dataset` and `source_type`. Handles random split."""

        random_ts_ids = all_ts_ids[self.ts_id_name]
        random_indices = np.arange(len(all_ts_ids))

        # Process train_ts if it was specified with times series ids
        if self.train_ts is not None and not isinstance(self.train_ts, (float, int)):
            self.train_ts, self.train_ts_row_ranges, _ = self._process_ts_ids(self.train_ts, all_ts_ids, all_ts_row_ranges, None, None)
            self.has_train = True

            mask = np.isin(random_ts_ids, self.train_ts, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("train_ts set: %s", self.train_ts)

        # Process val_ts if it was specified with times series ids
        if self.val_ts is not None and not isinstance(self.val_ts, (float, int)):
            self.val_ts, self.val_ts_row_ranges, _ = self._process_ts_ids(self.val_ts, all_ts_ids, all_ts_row_ranges, None, None)
            self.has_val = True

            mask = np.isin(random_ts_ids, self.val_ts, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("val_ts set: %s", self.val_ts)

        # Process time_ts if it was specified with times series ids
        if self.test_ts is not None and not isinstance(self.test_ts, (float, int)):
            self.test_ts, self.test_ts_row_ranges, _ = self._process_ts_ids(self.test_ts, all_ts_ids, all_ts_row_ranges, None, None)
            self.has_test = True

            mask = np.isin(random_ts_ids, self.test_ts, invert=True)
            random_ts_ids = random_ts_ids[mask]
            random_indices = random_indices[mask]

            self.logger.debug("test_ts set: %s", self.test_ts)

        # Convert proportions to total values
        if isinstance(self.train_ts, float):
            self.train_ts = int(self.train_ts * len(random_ts_ids))
            self.logger.debug("train_ts converted to total values: %s", self.train_ts)
        if isinstance(self.val_ts, float):
            self.val_ts = int(self.val_ts * len(random_ts_ids))
            self.logger.debug("val_ts converted to total values: %s", self.val_ts)
        if isinstance(self.test_ts, float):
            self.test_ts = int(self.test_ts * len(random_ts_ids))
            self.logger.debug("test_ts converted to total values: %s", self.test_ts)

        # Process random train_ts if it is to be randomly made
        if isinstance(self.train_ts, int):
            self.train_ts, self.train_ts_row_ranges, random_indices = self._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.train_ts, random_indices)
            self.has_train = True
            self.logger.debug("Random train_ts set with %s time series.", self.train_ts)

        # Process random val_ts if it is to be randomly made
        if isinstance(self.val_ts, int):
            self.val_ts, self.val_ts_row_ranges, random_indices = self._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.val_ts, random_indices)
            self.has_val = True
            self.logger.debug("Random val_ts set with %s time series.", self.val_ts)

        # Process random test_ts if it is to be randomly made
        if isinstance(self.test_ts, int):
            self.test_ts, self.test_ts_row_ranges, random_indices = self._process_ts_ids(None, all_ts_ids, all_ts_row_ranges, self.test_ts, random_indices)
            self.has_test = True
            self.logger.debug("Random test_ts set with %s time series.", self.test_ts)

        if not self.has_train and not self.has_val and not self.has_test:
            self.all_ts = all_ts_ids[self.ts_id_name]
            self.all_ts, self.all_ts_row_ranges, _ = self._process_ts_ids(self.all_ts, all_ts_ids, all_ts_row_ranges, None, None)
            self.logger.info("Using all time series for all_ts because train_ts, val_ts, and test_ts are all set to None.")
        else:
            for temp_ts_ids in [self.train_ts, self.val_ts, self.test_ts]:
                if temp_ts_ids is None:
                    continue
                elif self.all_ts is None:
                    self.all_ts = temp_ts_ids.copy()
                else:
                    self.all_ts = np.concatenate((self.all_ts, temp_ts_ids))

            if self.has_train:
                self.logger.debug("all_ts includes ids from train_ts.")
            if self.has_val:
                self.logger.debug("all_ts includes ids from val_ts.")
            if self.has_test:
                self.logger.debug("all_ts includes ids from test_ts.")

            self.all_ts, self.all_ts_row_ranges, _ = self._process_ts_ids(self.all_ts, all_ts_ids, all_ts_row_ranges, None, None)

        self.has_all = self.all_ts is not None

        if self.has_all:
            self.logger.debug("all_ts set with %s time series.", self.all_ts)

    def _set_feature_scalers(self) -> None:
        """Creates and/or validates scalers based on the `scale_with` parameter. """

        if self.scale_with is None:
            self.scale_with_display = None
            self.are_scalers_premade = False
            self.scalers = None
            self.is_scaler_custom = None

            self.logger.debug("No scaler will be used because scale_with is not set.")
            return

        # Treat scale_with as already initialized scaler
        if not isinstance(self.scale_with, (type, ScalerType)):
            self.scalers = self.scale_with

            if not self.has_train:
                if self.partial_fit_initialized_scalers:
                    self.logger.warning("partial_fit_initialized_scalers will be ignored because train set is not used.")
                self.partial_fit_initialized_scalers = False

            self.scale_with, self.scale_with_display = scaler_from_input_to_scaler_type(type(self.scale_with), check_for_fit=False, check_for_partial_fit=self.partial_fit_initialized_scalers)

            self.are_scalers_premade = True

            self.is_scaler_custom = "Custom" in self.scale_with_display
            self.logger.debug("Using initialized scaler of type: %s", self.scale_with_display)

        # Treat scale_with as uninitialized scaler
        else:
            if not self.has_train:
                self.scale_with = None
                self.scale_with_display = None
                self.are_scalers_premade = False
                self.scalers = None
                self.is_scaler_custom = None

                self.logger.warning("No scaler will be used because train set is not used.")
                return

            self.scale_with, self.scale_with_display = scaler_from_input_to_scaler_type(self.scale_with, check_for_fit=False, check_for_partial_fit=True)
            self.scalers = self.scale_with()

            self.are_scalers_premade = False

            self.is_scaler_custom = "Custom" in self.scale_with_display
            self.logger.debug("Using uninitialized scaler of type: %s", self.scale_with_display)

    def _set_fillers(self) -> None:
        """Creates and/or validates fillers based on the `fill_missing_with` parameter. """

        self.fill_missing_with, self.fill_missing_with_display = filler_from_input_to_type(self.fill_missing_with)
        self.is_filler_custom = "Custom" in self.fill_missing_with_display if self.fill_missing_with is not None else None

        if self.fill_missing_with is None:
            self.logger.debug("No filler is used because fill_missing_with is set to None.")
            return

        # Set the fillers for the training set
        if self.has_train:
            self.train_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.train_ts])
            self.logger.debug("Fillers for training set are set.")

        # Set the fillers for the validation set
        if self.has_val:
            self.val_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.val_ts])
            self.logger.debug("Fillers for validation set are set.")

        # Set the fillers for the test set
        if self.has_test:
            self.test_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.test_ts])
            self.logger.debug("Fillers for test set are set.")

        # Set the fillers for the all set
        if self.has_all:
            self.all_fillers = np.array([self.fill_missing_with(self.features_to_take_without_ids) for _ in self.all_ts])
            self.logger.debug("Fillers for all set are set.")

        self.logger.debug("Using filler: %s.", self.fill_missing_with_display)

    def _validate_finalization(self) -> None:
        """Performs final validation of the configuration. """

        train_size = 0
        if self.has_train:
            train_size += len(self.train_ts)

        val_size = 0
        if self.has_val:
            val_size += len(self.val_ts)

        test_size = 0
        if self.has_test:
            test_size += len(self.test_ts)

        # Check for overlap between train, val, and test sets
        if train_size + val_size + test_size > 0 and train_size + val_size + test_size != len(np.unique(self.all_ts)):
            self.logger.error("Overlap detected! Train, Val, and Test sets can't have the same IDs.")
            raise ValueError("Train, Val, and Test can't have the same IDs.")

    def __str__(self) -> str:

        if self.scale_with is None:
            scaler_part = f"Scaler type: {str(self.scale_with_display)}"
        else:
            scaler_part = f'''Scaler type: {str(self.scale_with_display)}
        Are scalers premade: {self.are_scalers_premade}
        Are premade scalers partial_fitted: {self.partial_fit_initialized_scalers}'''

        if self.include_time:
            time_part = f'''Time included: {str(self.include_time)}    
        Time format: {str(self.time_format)}'''
        else:
            time_part = f"Time included: {str(self.include_time)}"

        return f'''
Config Details:
    Used for database: {self.database_name}
    Aggregation: {str(self.aggregation)}
    Source: {str(self.source_type)}

    Time series
        Train time series IDS: {get_abbreviated_list_string(self.train_ts)}
        Val time series IDS: {get_abbreviated_list_string(self.val_ts)}
        Test time series IDS {get_abbreviated_list_string(self.test_ts)}
        All time series IDS {get_abbreviated_list_string(self.all_ts)}
    Time periods
        Time period: {str(self.display_time_period)}
    Features
        Taken features: {str(self.features_to_take_without_ids)}
        Default values: {self.default_values}
        Time series ID included: {str(self.include_ts_id)}
        {time_part}
    Fillers         
        Filler type: {str(self.fill_missing_with_display)}
    Scalers
        {scaler_part}
    Batch sizes
        Train batch size: {self.train_batch_size}
        Val batch size: {self.val_batch_size}
        Test batch size: {self.test_batch_size}
        All batch size: {self.all_batch_size}
    Default workers
        Train worker count: {str(self.train_workers)}
        Val worker count: {str(self.val_workers)}
        Test worker count: {str(self.test_workers)}
        All worker count: {str(self.all_workers)}
        Init worker count: {str(self.init_workers)}
    Other
        Nan threshold: {str(self.nan_threshold)}
        Random state: {self.random_state}
        Train dataloader order {str(self.train_dataloader_order)}
                '''
