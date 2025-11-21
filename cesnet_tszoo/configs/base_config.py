from typing import Literal
from numbers import Number
from abc import ABC, abstractmethod
import math
import logging

import numpy as np
import numpy.typing as npt

import cesnet_tszoo.version as version
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, MANDATORY_PREPROCESSES_ORDER, MANDATORY_PREPROCESSES_ORDER_ENUM
from cesnet_tszoo.utils.enums import FillerType, TimeFormat, TransformerType, DataloaderOrder, DatasetType, AnomalyHandlerType, PreprocessType
from cesnet_tszoo.utils.transformer import Transformer
from cesnet_tszoo.data_models.dataset_metadata import DatasetMetadata
from cesnet_tszoo.data_models.preprocess_note import PreprocessNote
from cesnet_tszoo.data_models.preprocess_order_group import PreprocessOrderGroup
import cesnet_tszoo.utils.transformer.factory as transformer_factories
import cesnet_tszoo.utils.filler.factory as filler_factories
import cesnet_tszoo.utils.anomaly_handler.factory as anomaly_handler_factories
import cesnet_tszoo.utils.custom_handler.factory as custom_handler_factories
from cesnet_tszoo.utils.custom_handler.factory import PerSeriesCustomHandlerFactory, AllSeriesCustomHandlerFactory, NoFitCustomHandlerFactory
import cesnet_tszoo.utils.css_styles.utils as css_utils
from cesnet_tszoo.data_models.holders import FillingHolder, AnomalyHandlerHolder, TransformerHolder, AllSeriesCustomHandlerHolder


class DatasetConfig(ABC):
    """
    Base class for configuration management. This class should **not** be used directly. Instead, use one of its derived classes, such as TimeBasedConfig, DisjointTimeBasedConfig or SeriesBasedConfig.

    For available configuration options, refer to [here][cesnet_tszoo.configs.base_config.DatasetConfig--configuration-options].

    Attributes:
        used_train_workers: Tracks the number of train workers in use. Helps determine if the train dataloader should be recreated based on worker changes.
        used_val_workers: Tracks the number of validation workers in use. Helps determine if the validation dataloader should be recreated based on worker changes.
        used_test_workers: Tracks the number of test workers in use. Helps determine if the test dataloader should be recreated based on worker changes.
        used_all_workers: Tracks the total number of all workers in use. Helps determine if the all dataloader should be recreated based on worker changes.
        import_identifier: Tracks the name of the config upon import. None if not imported.
        filler_factory: Represents factory used to create passed Filler type.
        anomaly_handler_factory: Represents factory used to create passed Anomaly Handler type.
        transformer_factory: Represents factory used to create passed Transformer type.
        can_fit_fillers: Whether fillers in this config, can be fitted.
        logger: Logger for displaying information. 

    The following attributes are initialized when CesnetDataset.set_dataset_config_and_initialize is called:

    Attributes:
        aggregation: The aggregation period used for the data.
        source_type: The source type of the data.
        database_name: Specifies which database this config applies to.
        features_to_take_without_ids: Features to be returned, excluding time or time series IDs.
        indices_of_features_to_take_no_ids: Indices of non-ID features in `features_to_take`.
        ts_id_name: Name of the time series ID, dependent on `source_type`.
        used_singular_train_time_series: Currently used singular train set time series for dataloader.
        used_singular_val_time_series: Currently used singular validation set time series for dataloader.
        used_singular_test_time_series: Currently used singular test set time series for dataloader.
        used_singular_all_time_series: Currently used singular all set time series for dataloader.       
        train_preprocess_order: All preprocesses used for train set. 
        val_preprocess_order: All preprocesses used for val set. 
        test_preprocess_order: All preprocesses used for test set. 
        all_preprocess_order: All preprocesses used for all set. 
        is_initialized: Flag indicating if the configuration has already been initialized. If true, config initialization will be skipped.  
        version: Version of cesnet-tszoo this config was made in.
        export_update_needed: Whether config was updated to newer version and should be exported.

    # Configuration options

    Attributes:
        features_to_take: Defines which features are used.
        default_values: Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature.
        train_batch_size: Batch size for the train dataloader, when window size is None.
        val_batch_size: Batch size for the validation dataloader, when window size is None.
        test_batch_size: Batch size for the test dataloader, when window size is None.
        all_batch_size: Batch size for the all dataloader, when window size is None.
        preprocess_order: Defines in which order preprocesses are used. Also can add to order a type of PerSeriesCustomHandler, AllSeriesCustomHandler or NoFitCustomHandler.
        fill_missing_with: Defines how to fill missing values in the dataset. Can pass enum [`FillerType`][cesnet_tszoo.utils.enums.FillerType] for built-in filler or pass a type of custom filler that must derive from [`Filler`][cesnet_tszoo.utils.filler.Filler] base class.
        transform_with: Defines the transformer to transform the dataset. Can pass enum [`TransformerType`][cesnet_tszoo.utils.enums.TransformerType] for built-in transformer, pass a type of custom transformer or instance of already fitted transformer(s).
        handle_anomalies_with: Defines the anomaly handler for handling anomalies in the dataset. Can pass enum [`AnomalyHandlerType`][cesnet_tszoo.utils.enums.AnomalyHandlerType] for built-in anomaly handler or a type of custom anomaly handler.
        partial_fit_initialized_transformers: If `True`, partial fitting on train set is performed when using initiliazed transformers.
        include_time: If `True`, time data is included in the returned values.
        include_ts_id: If `True`, time series IDs are included in the returned values.
        time_format: Format for the returned time data. When using TimeFormat.DATETIME, time will be returned as separate list along rest of the values.
        train_workers: Number of workers for loading training data. `0` means that the data will be loaded in the main process.
        val_workers: Number of workers for loading validation data. `0` means that the data will be loaded in the main process.
        test_workers: Number of workers for loading test data. `0` means that the data will be loaded in the main process.
        all_workers: Number of workers for loading all data. `0` means that the data will be loaded in the main process.
        init_workers: Number of workers for initial dataset processing during configuration. `0` means that the data will be loaded in the main process.
        nan_threshold: Maximum allowable percentage of missing data. Time series exceeding this threshold are excluded. Time series over the threshold will not be used. Used for `train/val/test/all` separately.
        create_transformer_per_time_series: If `True`, a separate transformer is created for each time series. Not used when using already initialized transformers. 
        dataset_type: Type of a dataset this config is used for.
        train_dataloader_order: Defines the order of data returned by the training dataloader.
        random_state: Fixes randomness for reproducibility during configuration and dataset initialization.              
    """

    def __init__(self,
                 features_to_take: list[str] | Literal["all"],
                 default_values: list[Number] | npt.NDArray[np.number] | dict[str, Number] | Number | Literal["default"] | None,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 all_batch_size: int,
                 preprocess_order: list[str, type],
                 fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None,
                 transform_with: type | TransformerType | list[Transformer] | np.ndarray[Transformer] | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None,
                 handle_anomalies_with: type | AnomalyHandlerType | Literal["z-score", "interquartile_range"] | None,
                 partial_fit_initialized_transformers: bool,
                 include_time: bool,
                 include_ts_id: bool,
                 time_format: TimeFormat | Literal["id_time", "datetime", "unix_time", "shifted_unix_time"],
                 train_workers: int,
                 val_workers: int,
                 test_workers: int,
                 all_workers: int,
                 init_workers: int,
                 nan_threshold: float,
                 create_transformer_per_time_series: bool,
                 dataset_type: DatasetType,
                 train_dataloader_order: DataloaderOrder | Literal["random", "sequential"],
                 random_state: int | None,
                 can_fit_fillers: bool,
                 logger: logging.Logger):

        self.used_train_workers = None
        self.used_val_workers = None
        self.used_test_workers = None
        self.used_all_workers = None
        self.import_identifier = None
        self.filler_factory = filler_factories.get_filler_factory(fill_missing_with)
        self.anomaly_handler_factory = anomaly_handler_factories.get_anomaly_handler_factory(handle_anomalies_with)
        self.transformer_factory = transformer_factories.get_transformer_factory(transform_with, create_transformer_per_time_series, partial_fit_initialized_transformers)
        self.can_fit_fillers = can_fit_fillers
        self.logger = logger

        self.aggregation = None
        self.source_type = None
        self.database_name = None
        self.features_to_take_without_ids = None
        self.indices_of_features_to_take_no_ids = None
        self.ts_id_name = None
        self.used_singular_train_time_series = None
        self.used_singular_val_time_series = None
        self.used_singular_test_time_series = None
        self.used_singular_all_time_series = None
        self.train_preprocess_order: list[PreprocessNote] = []
        self.val_preprocess_order: list[PreprocessNote] = []
        self.test_preprocess_order: list[PreprocessNote] = []
        self.all_preprocess_order: list[PreprocessNote] = []
        self.is_initialized = False
        self.version = version.current_version
        self.export_update_needed = False

        self.features_to_take = features_to_take
        self.default_values = default_values
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.all_batch_size = all_batch_size
        self.preprocess_order = list(preprocess_order)
        self.partial_fit_initialized_transformers = partial_fit_initialized_transformers
        self.include_time = include_time
        self.include_ts_id = include_ts_id
        self.time_format = time_format
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers
        self.all_workers = all_workers
        self.init_workers = init_workers
        self.nan_threshold = nan_threshold
        self.create_transformer_per_time_series = create_transformer_per_time_series
        self.dataset_type = dataset_type
        self.train_dataloader_order = train_dataloader_order
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(random_state)

        self._validate_construction()

        self.logger.info("Quick validation succeeded.")

    def _validate_construction(self) -> None:
        """Performs basic parameter validation to ensure correct configuration. More comprehensive validation, which requires dataset-specific data, is handled in [`_dataset_init`][cesnet_tszoo.configs.base_config.DatasetConfig._dataset_init]. """

        # Ensuring boolean flags are correctly set
        assert isinstance(self.partial_fit_initialized_transformers, bool), "partial_fit_initialized_transformers must be a boolean value."
        assert isinstance(self.include_time, bool), "include_time must be a boolean value."
        assert isinstance(self.include_ts_id, bool), "include_ts_id must be a boolean value."
        assert isinstance(self.create_transformer_per_time_series, bool), "create_transformer_per_time_series must be a boolean value."

        # Ensuring worker count values are non-negative integers
        assert isinstance(self.train_workers, int) and self.train_workers >= 0, "train_workers must be a non-negative integer."
        assert isinstance(self.val_workers, int) and self.val_workers >= 0, "val_workers must be a non-negative integer."
        assert isinstance(self.test_workers, int) and self.test_workers >= 0, "test_workers must be a non-negative integer."
        assert isinstance(self.all_workers, int) and self.all_workers >= 0, "all_workers must be a non-negative integer."
        assert isinstance(self.init_workers, int) and self.init_workers >= 0, "init_workers must be a non-negative integer."

        # Ensuring batch size values are positive integers
        assert isinstance(self.train_batch_size, int) and self.train_batch_size > 0, "train_batch_size must be a positive integer."
        assert isinstance(self.val_batch_size, int) and self.val_batch_size > 0, "val_batch_size must be a positive integer."
        assert isinstance(self.test_batch_size, int) and self.test_batch_size > 0, "test_batch_size must be a positive integer."
        assert isinstance(self.all_batch_size, int) and self.all_batch_size > 0, "all_batch_size must be a positive integer."

        # Ensuring that preprocess order contains all required preprocesses
        assert self.preprocess_order is not None, "preprocess_order must be set."
        assert isinstance(self.preprocess_order, list), "preprocess_order must be list"
        assert MANDATORY_PREPROCESSES_ORDER.issubset(self.preprocess_order) or MANDATORY_PREPROCESSES_ORDER_ENUM.issubset(self.preprocess_order), f"preprocess_order must at least contain order for {list(MANDATORY_PREPROCESSES_ORDER)}"

        mandatory_count = 0
        for preprocess in self.preprocess_order:
            if isinstance(preprocess, (str, PreprocessType)):
                PreprocessType(preprocess)
                mandatory_count += 1
            elif not isinstance(preprocess, type):
                raise ValueError(f"Values in preprocess_order must be either from {list(MANDATORY_PREPROCESSES_ORDER)} or a type.")

        if mandatory_count != len(MANDATORY_PREPROCESSES_ORDER):
            raise ValueError(f"preprocess_order must not contain duplicate mandatory preprocesses ({MANDATORY_PREPROCESSES_ORDER}).")

        # Validate nan_threshold value
        assert isinstance(self.nan_threshold, Number) and 0 <= self.nan_threshold <= 1, "nan_threshold must be a number between 0 and 1."
        self.nan_threshold = float(self.nan_threshold)

        # Convert time_format and train_dataloader_order to their respective enum types
        self.time_format = TimeFormat(self.time_format)
        self.train_dataloader_order = DataloaderOrder(self.train_dataloader_order)

    def _update_batch_sizes(self, train_batch_size: int, val_batch_size: int, test_batch_size: int, all_batch_size: int) -> None:

        # Ensuring batch size values are positive integers
        assert isinstance(train_batch_size, int) and train_batch_size > 0, "train_batch_size must be a positive integer."
        assert isinstance(val_batch_size, int) and val_batch_size > 0, "val_batch_size must be a positive integer."
        assert isinstance(test_batch_size, int) and test_batch_size > 0, "test_batch_size must be a positive integer."
        assert isinstance(all_batch_size, int) and all_batch_size > 0, "all_batch_size must be a positive integer."

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.all_batch_size = all_batch_size

        self.logger.debug("Updated batch sizes.")

    def _update_workers(self, train_workers: int, val_workers: int, test_workers: int, all_workers: int, init_workers: int) -> None:

        # Ensuring worker count values are non-negative integers
        assert isinstance(self.train_workers, int) and self.train_workers >= 0, "train_workers must be a non-negative integer."
        assert isinstance(self.val_workers, int) and self.val_workers >= 0, "val_workers must be a non-negative integer."
        assert isinstance(self.test_workers, int) and self.test_workers >= 0, "test_workers must be a non-negative integer."
        assert isinstance(self.all_workers, int) and self.all_workers >= 0, "all_workers must be a non-negative integer."
        assert isinstance(self.init_workers, int) and self.init_workers >= 0, "init_workers must be a non-negative integer."

        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers
        self.all_workers = all_workers
        self.init_workers = init_workers

        self.logger.debug("Updated workers.")

    @abstractmethod
    def _get_train(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the training set. """
        ...

    @abstractmethod
    def _get_val(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the validation set. """
        ...

    @abstractmethod
    def _get_test(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the test set. """
        ...

    @abstractmethod
    def _get_all(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Returns the indices corresponding to the all set. """
        ...

    @abstractmethod
    def has_train(self) -> bool:
        """Returns whether training set is used. """
        ...

    @abstractmethod
    def has_val(self) -> bool:
        """Returns whether validation set is used. """
        ...

    @abstractmethod
    def has_test(self) -> bool:
        """Returns whether test set is used. """
        ...

    @abstractmethod
    def has_all(self) -> bool:
        """Returns whether all set is used. """
        ...

    def _get_train_preprocess_init_order_groups(self) -> list[PreprocessOrderGroup]:
        return self.__get_preprocess_init_order_groups(self.train_preprocess_order)

    def _get_val_preprocess_init_order_groups(self) -> list[PreprocessOrderGroup]:
        return self.__get_preprocess_init_order_groups(self.val_preprocess_order)

    def _get_test_preprocess_init_order_groups(self) -> list[PreprocessOrderGroup]:
        return self.__get_preprocess_init_order_groups(self.test_preprocess_order)

    def __get_preprocess_init_order_groups(self, preprocess_order) -> list[PreprocessOrderGroup]:
        """Returns preprocess grouped orders used when initializing config. """

        groups = []

        outers = []
        inners = []

        preprocess_note: PreprocessNote
        for preprocess_note in preprocess_order:

            if preprocess_note.is_inner_preprocess:

                if len(outers) > 0:
                    group = PreprocessOrderGroup(inners + outers)
                    groups.append(group)

                    inners = group.get_preprocess_orders_for_inner_transform()
                    outers.clear()

                inners.append(preprocess_note)

            if not preprocess_note.is_inner_preprocess:
                outers.append(preprocess_note)

        group = PreprocessOrderGroup(inners + outers)

        if group.any_preprocess_needs_fitting or group.any_preprocess_is_dummy_fitting:
            groups.append(group)

        if len(groups) == 0:
            groups.append(PreprocessOrderGroup([]))

        return groups

    def _update_identifiers_from_dataset_metadata(self, dataset_metadata: DatasetMetadata) -> None:
        """Updates identifying attributes from dataset metadata. """

        self.aggregation = dataset_metadata.aggregation
        self.source_type = dataset_metadata.source_type
        self.database_name = dataset_metadata.database_name

    def _dataset_init(self, dataset_metadata: DatasetMetadata) -> None:
        """Performs deeper parameter validation and updates values based on data from the dataset. """

        self.ts_id_name = dataset_metadata.ts_id_name

        self._set_features_to_take(dataset_metadata.features)
        self.logger.debug("Features to take have been successfully set.")

        self._set_ts(dataset_metadata.ts_indices, dataset_metadata.ts_row_ranges)
        self.logger.debug("Time series IDs have been successfully set.")

        self._set_time_period(dataset_metadata.time_indices)
        self.logger.debug("Time period have been successfully set.")

        self._set_default_values(dataset_metadata.default_values)
        self.logger.debug("Default values have been successfully set.")

        self._set_preprocess_order()
        self.logger.debug("Preprocess order have been successfully set.")

        self._validate_finalization()
        self.logger.debug("Finalization and validation completed successfully.")

    def _set_features_to_take(self, all_dataset_features: dict[str, np.dtype]) -> None:
        """Validates and filters the input `features_to_take` based on the `dataset`, `source_type`, and `aggregation`. """

        if self.features_to_take == "all":
            self.features_to_take = list(all_dataset_features.keys())
            self.logger.debug("All features used because 'features_to_take' is set to 'all'.")

        # Handling the inclusion of time ID in features
        if self.include_time and self.features_to_take.count(ID_TIME_COLUMN_NAME) == 0 and self.time_format != TimeFormat.DATETIME:
            self.features_to_take.insert(0, ID_TIME_COLUMN_NAME)
            self.logger.debug("Added '%s' to the features as 'include_time' is true and 'time_format' is not datetime.", ID_TIME_COLUMN_NAME)
        elif self.include_time and self.features_to_take.count(ID_TIME_COLUMN_NAME) > 0 and self.time_format == TimeFormat.DATETIME:
            self.features_to_take.remove(ID_TIME_COLUMN_NAME)
            self.logger.debug("Removed '%s' from the features because 'time_format' is datetime.", ID_TIME_COLUMN_NAME)
        elif not self.include_time and self.features_to_take.count(ID_TIME_COLUMN_NAME) > 0:
            self.features_to_take.remove(ID_TIME_COLUMN_NAME)
            self.logger.debug("Removed '%s' from the features as 'include_time' is false.", ID_TIME_COLUMN_NAME)

        # Handling the inclusion of time series ID feature
        if self.include_ts_id and self.features_to_take.count(self.ts_id_name) <= 0:
            self.features_to_take.insert(0, self.ts_id_name)
            self.logger.debug("Added '%s' to the features as 'include_ts_id' is true.", self.ts_id_name)
        elif not self.include_ts_id and self.features_to_take.count(self.ts_id_name) > 0:
            self.features_to_take.remove(self.ts_id_name)
            self.logger.debug("Removed '%s' from the features as 'include_ts_id' is false.", self.ts_id_name)

        # Filtering features based on available dataset features
        temp = list(self.features_to_take)
        self.features_to_take = [feature for feature in self.features_to_take if feature in all_dataset_features or feature == ID_TIME_COLUMN_NAME or feature == self.ts_id_name]

        if len(temp) != len(self.features_to_take):
            self.logger.warning("Some features were removed as they are not available in the dataset.")

        # Preparing indices and features without time and time series ID
        self.indices_of_features_to_take_no_ids = [idx for idx, feature in enumerate(self.features_to_take) if feature != ID_TIME_COLUMN_NAME and feature != self.ts_id_name]
        self.features_to_take_without_ids = [feature for feature in self.features_to_take if feature != ID_TIME_COLUMN_NAME and feature != self.ts_id_name]

        # Assert that at least one feature is used
        assert len(self.features_to_take_without_ids) > 0, "At least one non-ID feature must be used."

    def _set_default_values(self, default_values: dict[str, Number]) -> None:
        """Validates and filters the input `default_values` based on the `dataset`, `source_type`, `aggregation`, and `features_to_take`. """

        if self.default_values == "default":
            self.default_values = dict(default_values)
            self.logger.debug("Using default dataset values for default values because 'default_values' is set to 'default'.")

        elif isinstance(self.default_values, Number):
            # If default_values is a single number, assign it to all features

            orig_default_value = self.default_values
            self.default_values = {feature: float(self.default_values) for feature in self.features_to_take_without_ids}
            self.logger.debug("Assigned the default value %s to all features as 'default_values' is a single number.", float(orig_default_value))

        elif isinstance(self.default_values, (list, np.ndarray)):
            # If default_values is a list or ndarray, ensure the length matches with features_to_take_without_ids
            if len(self.default_values) != len(self.features_to_take_without_ids):
                raise ValueError("The number of values in 'default_values' does not match the number of features in 'features_to_take'.")
            self.default_values = {feature: value for feature, value in zip(self.features_to_take_without_ids, self.default_values) if feature != ID_TIME_COLUMN_NAME and feature != self.ts_id_name}
            self.logger.debug("Mapped default values to features, skipping IDs features: %s", self.default_values)

        elif isinstance(self.default_values, dict):
            # If default_values is a dictionary, ensure its keys match the features
            if set(self.default_values.keys()) != set(self.features_to_take_without_ids):
                raise ValueError("The keys in 'default_values' do not match the features in 'features_to_take'.")
            self.logger.debug("Using provided default values for features: %s", self.default_values)

        elif self.default_values is None or math.isnan(self.default_values) or np.isnan(self.default_values):
            # If default_values is None or NaN, assign NaN to each feature
            self.default_values = {feature: np.nan for feature in self.features_to_take_without_ids}
            self.logger.debug("Assigned NaN as the default value for all features because 'default_values' is None or NaN.")

        # Convert the default values into a NumPy array for consistent data handling
        temp_default_values = np.ndarray(len(self.features_to_take_without_ids), np.float64)
        for i, feature in enumerate(self.features_to_take_without_ids):
            temp_default_values[i] = self.default_values[feature]

        self.default_values = temp_default_values

    def _set_preprocess_order(self):
        """Validates and converts preprocess order to their enum variant. Also initializes preprocess_orders for all sets. """

        for i, order in enumerate(self.preprocess_order):
            if isinstance(order, (str, PreprocessType)):
                self.preprocess_order[i] = PreprocessType(order)
            elif not isinstance(order, type):
                raise NotImplementedError("Currenty preprocess order supports only string names or types")

        self._init_preprocess_order()

    def _init_preprocess_order(self):
        self.train_preprocess_order = []
        self.val_preprocess_order = []
        self.test_preprocess_order = []
        self.all_preprocess_order = []

        for preprocess_type in self.preprocess_order:

            if preprocess_type == PreprocessType.TRANSFORMING:
                self.__set_transform_order(preprocess_type)
            elif preprocess_type == PreprocessType.FILLING_GAPS:
                self.__set_filling_order(preprocess_type)
            elif preprocess_type == PreprocessType.HANDLING_ANOMALIES:
                self.__set_anomaly_handler_order(preprocess_type)
            elif isinstance(preprocess_type, type):
                self.__set_custom_handler(preprocess_type)
            else:
                raise NotImplementedError()

    def __set_transform_order(self, preprocess_type: PreprocessType):
        needs_fitting = (self.partial_fit_initialized_transformers or not self.transformer_factory.has_already_initialized) and not self.transformer_factory.is_empty_factory
        should_partial_fit = (self.transformer_factory.has_already_initialized and self.partial_fit_initialized_transformers) or (not self.transformer_factory.has_already_initialized and not self.create_transformer_per_time_series)
        is_outer = not self.create_transformer_per_time_series and needs_fitting
        transformers = self._get_feature_transformers()

        self.train_preprocess_order.append(PreprocessNote(preprocess_type, False, needs_fitting, self.has_train(), not is_outer, TransformerHolder(transformers, self.create_transformer_per_time_series, should_partial_fit)))
        self.val_preprocess_order.append(PreprocessNote(preprocess_type, needs_fitting, False, self.has_val(), not is_outer, TransformerHolder(transformers, self.create_transformer_per_time_series, False)))
        self.test_preprocess_order.append(PreprocessNote(preprocess_type, needs_fitting, False, self.has_test(), not is_outer, TransformerHolder(transformers, self.create_transformer_per_time_series, False)))
        self.all_preprocess_order.append(PreprocessNote(preprocess_type, needs_fitting, False, self.has_all(), not is_outer, TransformerHolder(transformers, self.create_transformer_per_time_series, False)))

    def __set_filling_order(self, preprocess_type: PreprocessType):
        needs_fitting = self.can_fit_fillers and not self.filler_factory.is_empty_factory
        train_fillers, val_fillers, test_fillers, all_fillers = self._get_fillers()

        self.train_preprocess_order.append(PreprocessNote(preprocess_type, needs_fitting, False, self.has_train(), True, FillingHolder(train_fillers, self.default_values)))
        self.val_preprocess_order.append(PreprocessNote(preprocess_type, False, needs_fitting, self.has_val(), True, FillingHolder(val_fillers, self.default_values)))
        self.test_preprocess_order.append(PreprocessNote(preprocess_type, False, needs_fitting, self.has_test(), True, FillingHolder(test_fillers, self.default_values)))
        self.all_preprocess_order.append(PreprocessNote(preprocess_type, needs_fitting, False, self.has_all(), True, FillingHolder(all_fillers, self.default_values)))

    def __set_anomaly_handler_order(self, preprocess_type: PreprocessType):
        anomaly_handlers = self._get_anomaly_handlers()

        self.train_preprocess_order.append(PreprocessNote(preprocess_type, False, not self.anomaly_handler_factory.is_empty_factory, self.has_train(), True, AnomalyHandlerHolder(anomaly_handlers)))
        self.val_preprocess_order.append(PreprocessNote(preprocess_type, not self.anomaly_handler_factory.is_empty_factory, False, self.has_val(), True, AnomalyHandlerHolder(None)))
        self.test_preprocess_order.append(PreprocessNote(preprocess_type, not self.anomaly_handler_factory.is_empty_factory, False, self.has_test(), True, AnomalyHandlerHolder(None)))
        self.all_preprocess_order.append(PreprocessNote(preprocess_type, not self.anomaly_handler_factory.is_empty_factory, False, self.has_all(), True, AnomalyHandlerHolder(None)))

    def _update_preprocess_order_supported_ids(self, preprocess_order: list[PreprocessNote], supported_ts_ids: np.ndarray | list):
        for preprocess in preprocess_order:
            preprocess.holder.supported_ts_updated(supported_ts_ids)

    def __set_custom_handler(self, preprocess_type: type):
        factory = custom_handler_factories.get_custom_handler_factory(preprocess_type)

        if factory.preprocess_enum_type == PreprocessType.PER_SERIES_CUSTOM:
            self._set_per_series_custom_handler(factory)
        elif factory.preprocess_enum_type == PreprocessType.ALL_SERIES_CUSTOM:
            self.__set_all_series_custom_handler(factory)
        elif factory.preprocess_enum_type == PreprocessType.NO_FIT_CUSTOM:
            self._set_no_fit_custom_handler(factory)
        else:
            raise NotImplementedError()

    @abstractmethod
    def _set_per_series_custom_handler(self, factory: PerSeriesCustomHandlerFactory):
        ...

    def __set_all_series_custom_handler(self, factory: AllSeriesCustomHandlerFactory):

        if not self.has_train():
            raise ValueError("To use AllSeriesCustomHandlerFactory you need to use train set.")

        handler = factory.create_handler()

        self.train_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, False, True, factory.can_apply_to_train, False, AllSeriesCustomHandlerHolder(handler)))
        self.val_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, True, False, factory.can_apply_to_val and self.has_val(), False, AllSeriesCustomHandlerHolder(handler)))
        self.test_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, True, False, factory.can_apply_to_test and self.has_test(), False, AllSeriesCustomHandlerHolder(handler)))
        self.all_preprocess_order.append(PreprocessNote(factory.preprocess_enum_type, True, False, factory.can_apply_to_all and self.has_all(), False, AllSeriesCustomHandlerHolder(handler)))

    def _get_summary_steps(self) -> list[css_utils.SummaryDiagramStep]:
        steps = []

        steps.append(self._get_summary_dataset())
        steps.append(self._get_summary_filter_time_series())
        steps.append(self._get_summary_filter_features())
        steps += self._get_summary_preprocessing()
        steps += self._get_summary_loader()

        return steps

    def _get_summary_dataset(self) -> css_utils.SummaryDiagramStep:
        attributes = [css_utils.StepAttribute("Database", self.database_name),
                      css_utils.StepAttribute("Aggregation", self.aggregation),
                      css_utils.StepAttribute("Source", self.source_type)]

        return css_utils.SummaryDiagramStep("Load from dataset", attributes)

    @abstractmethod
    def _get_summary_filter_time_series(self) -> css_utils.SummaryDiagramStep:
        ...

    def _get_summary_filter_features(self) -> css_utils.SummaryDiagramStep:
        attributes = [css_utils.StepAttribute("Taken features", self.features_to_take_without_ids),
                      css_utils.StepAttribute("Time series ID included", self.include_ts_id),
                      css_utils.StepAttribute("Time included", self.include_time),
                      css_utils.StepAttribute("Time format", self.time_format)]

        return css_utils.SummaryDiagramStep("Filter features", attributes)

    def _get_summary_preprocessing(self) -> list[css_utils.SummaryDiagramStep]:
        steps = []

        for preprocess_type, train_pr, val_pr, test_pr, all_pr in list(zip(self.preprocess_order, self.train_preprocess_order, self.val_preprocess_order, self.test_preprocess_order, self.all_preprocess_order)):
            preprocess_title = None
            preprocess_type_name = None
            is_per_time_series = train_pr.is_inner_preprocess
            target_sets = []
            requires_fitting = False
            if train_pr.can_be_applied:
                target_sets.append("train")
                requires_fitting = train_pr.should_be_fitted
            if val_pr.can_be_applied:
                target_sets.append("val")
            if test_pr.can_be_applied:
                target_sets.append("test")
            if all_pr.can_be_applied:
                target_sets.append("all")

            if len(target_sets) == 0:
                continue

            if train_pr.preprocess_type == PreprocessType.HANDLING_ANOMALIES:
                preprocess_title = "Handle anomalies"
                preprocess_type_name = self.anomaly_handler_factory.anomaly_handler_type.__name__

                if self.anomaly_handler_factory.is_empty_factory:
                    continue

            elif train_pr.preprocess_type == PreprocessType.FILLING_GAPS:
                preprocess_title = "Handle missing values"
                preprocess_type_name = f"{self.filler_factory.filler_type.__name__}"

                steps.append(css_utils.SummaryDiagramStep("Pre-fill with default values", [css_utils.StepAttribute("Default values", self.default_values)]))

                if self.filler_factory.is_empty_factory:
                    continue

            elif train_pr.preprocess_type == PreprocessType.TRANSFORMING:
                preprocess_title = "Apply transformer"
                preprocess_type_name = self.transformer_factory.transformer_type.__name__

                if self.transformer_factory.is_empty_factory:
                    continue

                is_per_time_series = self.create_transformer_per_time_series
            elif train_pr.preprocess_type == PreprocessType.PER_SERIES_CUSTOM:
                preprocess_title = f"Apply {preprocess_type.__name__}"
                preprocess_type_name = preprocess_type.__name__
            elif train_pr.preprocess_type == PreprocessType.ALL_SERIES_CUSTOM:
                preprocess_title = f"Apply {preprocess_type.__name__}"
                preprocess_type_name = preprocess_type.__name__
            elif train_pr.preprocess_type == PreprocessType.NO_FIT_CUSTOM:
                preprocess_title = f"Apply {preprocess_type.__name__}"
                preprocess_type_name = preprocess_type.__name__

            step = css_utils.SummaryDiagramStep(preprocess_title, [css_utils.StepAttribute("Type", preprocess_type_name),
                                                                   css_utils.StepAttribute("Requires fitting", requires_fitting),
                                                                   css_utils.StepAttribute("Is per time series", is_per_time_series),
                                                                   css_utils.StepAttribute("Target sets", target_sets)])
            steps.append(step)

        return steps

    @abstractmethod
    def _get_summary_loader(self) -> list[css_utils.SummaryDiagramStep]:
        ...

    @abstractmethod
    def _set_no_fit_custom_handler(self, factory: NoFitCustomHandlerFactory):
        ...

    @abstractmethod
    def _set_time_period(self, all_time_ids: np.ndarray) -> None:
        """Validates and filters the input time periods based on the dataset and aggregation. This typically calls [`_process_time_period`][cesnet_tszoo.configs.base_config.DatasetConfig._process_time_period] for each time period. """
        ...

    @abstractmethod
    def _set_ts(self, all_ts_ids: np.ndarray, all_ts_row_ranges: np.ndarray) -> None:
        """Validates and filters the input time series IDs based on the `dataset` and `source_type`. This typically calls [`_process_ts_ids`][cesnet_tszoo.configs.base_config.DatasetConfig._process_ts_ids] for each time series ID filter. """
        ...

    @abstractmethod
    def _get_feature_transformers(self) -> np.ndarray[Transformer] | Transformer:
        """Creates transformers with `transformer_factory`. """
        ...

    @abstractmethod
    def _get_fillers(self) -> tuple:
        """Creates fillers with `filler_factory`. """
        ...

    @abstractmethod
    def _get_anomaly_handlers(self) -> np.ndarray:
        """Creates anomaly handlers with `anomaly_handler_factory`. """
        ...

    @abstractmethod
    def _validate_finalization(self) -> None:
        """Performs final validation of the configuration. """
        ...
