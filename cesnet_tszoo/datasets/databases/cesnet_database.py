import logging
import os
from abc import ABC

import cesnet_tszoo.datasets.dataset_factory as dataset_factories
from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType
from cesnet_tszoo.utils.download import resumable_download
from cesnet_tszoo.data_models.dataset_metadata import DatasetMetadata


class CesnetDatabase(ABC):
    """
    Base class for cesnet databases. This class should **not** be used directly. Use it as base for adding new databases.

    Derived databases are used by calling class method [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset] which will create a new instance of specified CesnetDataset. Check them for more info about how to use them.

    **Intended usage:**

    When using [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset] (`dataset_type` = `DatasetType.TIME_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy].     

    When using [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] (`dataset_type` = `DatasetType.SERIES_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_numpy].   

    When using [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset] (`dataset_type` = `DatasetType.DISJOINT_TIME_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`DisjointTimeBasedConfig`][cesnet_tszoo.configs.disjoint_time_based_config.DisjointTimeBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_numpy].   

    Used class attributes:

    Attributes:
        name: Name of the database.
        bucket_url: URL of the bucket where the dataset is stored.
        tszoo_root: Path to folder where all databases are saved. Set after [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset] was called at least once.
        database_root: Path to the folder where datasets belonging to the database are saved. Set after [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset] was called at least once.
        configs_root: Path to the folder where configurations are saved. Set after [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset] was called at least once.
        benchmarks_root: Path to the folder where benchmarks are saved. Set after [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset] was called at least once.
        annotations_root: Path to the folder where annotations are saved. Set after [`get_dataset`][cesnet_tszoo.datasets.databases.CesnetDatabase.get_dataset] was called at least once.
        id_names: Names for time series IDs for each `source_type`.
        default_values: Default values for each available feature.
        source_types: Available source types for the database.
        aggregations: Available aggregations for the database.   
        additional_data: Available small datasets for each dataset. 
    """

    name: str
    bucket_url: str

    tszoo_root: str
    database_root: str
    configs_root: str
    benchmarks_root: str
    annotations_root: str

    id_names: dict = None
    default_values: dict = None
    source_types: list[SourceType] = []
    aggregations: list[AgreggationType] = []
    additional_data: dict[str, tuple] = {}

    def __init__(self):
        raise ValueError("To create dataset instance use class method 'get_dataset' instead.")

    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | str, aggregation: AgreggationType | str, dataset_type: DatasetType | str, check_errors: bool = False, display_details: bool = False) -> CesnetDataset:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            source_type: The source type of the desired dataset.
            aggregation: The aggregation type for the selected source type.
            dataset_type: Type of a dataset you want to create. Can be [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset], [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] or [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset].
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`

        Returns:
            [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset], [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] or [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset].
        """

        logger = logging.getLogger("wrapper_dataset")

        source_type = SourceType(source_type)
        aggregation = AgreggationType(aggregation)
        dataset_type = DatasetType(dataset_type)

        if source_type not in cls.source_types:
            raise ValueError(f"Unsupported source type: {source_type}")

        if aggregation not in cls.aggregations:
            raise ValueError(f"Unsupported aggregation type: {aggregation}")

        # Dataset paths setup
        cls.tszoo_root = os.path.normpath(os.path.expanduser(os.path.join(data_root, "tszoo")))
        cls.database_root = os.path.join(cls.tszoo_root, "databases", cls.name)
        cls.configs_root = os.path.join(cls.tszoo_root, "configs")
        cls.benchmarks_root = os.path.join(cls.tszoo_root, "benchmarks")
        cls.annotations_root = os.path.join(cls.tszoo_root, "annotations")
        dataset_name = f"{cls.name}-{source_type.value}-{AgreggationType._to_str_without_number(aggregation)}"
        dataset_path = os.path.join(cls.database_root, f"{dataset_name}.h5")

        # Ensure necessary directories exist
        for directory in [cls.database_root, cls.configs_root, cls.annotations_root, cls.benchmarks_root]:
            if not os.path.exists(directory):
                logger.info("Creating directory: %s", directory)
                os.makedirs(directory)

        if not cls._is_downloaded(dataset_path):
            cls._download(dataset_name, dataset_path)

        dataset_metadata = DatasetMetadata(cls.name, dataset_type, dataset_path, cls.configs_root, cls.benchmarks_root, cls.annotations_root, source_type, aggregation, cls.id_names[source_type], cls.default_values, cls.additional_data)

        dataset_factory = dataset_factories.get_dataset_factory(dataset_type)
        dataset = dataset_factory.create_dataset(dataset_metadata)

        if check_errors:
            dataset.check_errors()

        if display_details:
            dataset.display_dataset_details()

        return dataset

    @classmethod
    def get_expected_paths(cls, data_root: str, database_name: str) -> dict:
        """Returns expected path for the provided `data_root` and `database_name`

        Args:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            database_name: Name of the expected database.

        Returns:
            str: Dictionary of paths.
        """

        paths = {}

        paths["tszoo_root"] = os.path.normpath(os.path.expanduser(os.path.join(data_root, "tszoo")))
        paths["database_root"] = os.path.join(paths["tszoo_root"], "databases", database_name)
        paths["configs_root"] = os.path.join(paths["tszoo_root"], "configs")
        paths["benchmarks_root"] = os.path.join(paths["tszoo_root"], "benchmarks")
        paths["annotations_root"] = os.path.join(paths["tszoo_root"], "annotations")

        return paths

    @classmethod
    def _is_downloaded(cls, dataset_path: str) -> bool:
        """Check whether the dataset at path has already been downloaded. """

        return os.path.exists(dataset_path)

    @classmethod
    def _download(cls, dataset_name: str, dataset_path: str) -> None:
        """Download the dataset file. """

        logger = logging.getLogger("wrapper_dataset")

        logger.info("Downloading %s dataset.", dataset_name)
        database_url = f"{cls.bucket_url}&file={dataset_name}.h5"
        resumable_download(url=database_url, file_path=dataset_path, silent=False)
