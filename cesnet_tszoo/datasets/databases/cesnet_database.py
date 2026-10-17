import logging
import os
from abc import ABC

import cesnet_tszoo.datasets.dataset_factory as dataset_factories
from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType, DownloadSource
from cesnet_tszoo.utils.download import download_from_sources, source_kind
from cesnet_tszoo.data_models.dataset_metadata import DatasetMetadata


class CesnetDatabase(ABC):
    """
    Base class for cesnet databases. This class should **not** be used directly. Use it as base for adding new databases.

    Derived databases are used by calling class method [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset) which will create a new instance of specified CesnetDataset. Check them for more info about how to use them.

    **Intended usage:**

    When using [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset) (`dataset_type` = `DatasetType.TIME_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset). This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`TimeBasedConfig`](reference_time_based_config.md#references.TimeBasedConfig) and set it using [`set_dataset_config_and_initialize`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize). 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset)/[`get_train_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df)/[`get_train_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy) to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader)/[`get_val_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df)/[`get_val_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy).
    5. Evaluate the model on [`get_test_dataloader`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader)/[`get_test_df`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df)/[`get_test_numpy`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy).     

    When using [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) (`dataset_type` = `DatasetType.SERIES_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset). This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`SeriesBasedConfig`](reference_series_based_config.md#references.SeriesBasedConfig) and set it using [`set_dataset_config_and_initialize`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize). 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_dataloader)/[`get_train_df`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_df)/[`get_train_numpy`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_numpy) to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_dataloader)/[`get_val_df`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_df)/[`get_val_numpy`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_numpy).
    5. Evaluate the model on [`get_test_dataloader`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_dataloader)/[`get_test_df`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_df)/[`get_test_numpy`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_numpy).   

    When using [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset) (`dataset_type` = `DatasetType.DISJOINT_TIME_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset). This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`DisjointTimeBasedConfig`](reference_disjoint_time_based_config.md#references.DisjointTimeBasedConfig) and set it using [`set_dataset_config_and_initialize`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize). 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_dataloader)/[`get_train_df`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_df)/[`get_train_numpy`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_numpy) to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_dataloader)/[`get_val_df`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_df)/[`get_val_numpy`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_numpy).
    5. Evaluate the model on [`get_test_dataloader`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_dataloader)/[`get_test_df`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_df)/[`get_test_numpy`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_numpy).   

    Used class attributes:

    Attributes:
        name: Name of the database.
        bucket_url: URL of the CESNET DataZoo bucket where the dataset is stored (`None` when the database is not in DataZoo). Used as the last download source.
        download_urls: Additional download sources per dataset file name (`dataset_name` -> list of sources). A source is a plain URL or a multi-part description from [`parts_source`][cesnet_tszoo.utils.download.parts_source] (file split into consecutive parts, optionally gzipped, e.g. on Zenodo). Mirrors are tried in order, before `bucket_url`. Use [`google_drive_url`][cesnet_tszoo.utils.download.google_drive_url] / [`zenodo_file_url`][cesnet_tszoo.utils.download.zenodo_file_url] to build the URLs.
        fallback_download_urls: Secondary download sources per dataset file name, in the same format as `download_urls` but tried *after* `bucket_url`. Use these for a database published in DataZoo that only needs a backup mirror; the primary source stays DataZoo.
        checksums: Optional SHA-256 hex digests per dataset file name (`dataset_name` -> digest). When set, the downloaded file is verified and a corrupted download is retried from the next source.
        tszoo_root: Path to folder where all databases are saved. Set after [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset) was called at least once.
        database_root: Path to the folder where datasets belonging to the database are saved. Set after [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset) was called at least once.
        configs_root: Path to the folder where configurations are saved. Set after [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset) was called at least once.
        benchmarks_root: Path to the folder where benchmarks are saved. Set after [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset) was called at least once.
        annotations_root: Path to the folder where annotations are saved. Set after [`get_dataset`](reference_cesnet_database.md#cesnet_tszoo.datasets.databases.cesnet_database.CesnetDatabase.get_dataset) was called at least once.
        id_names: Names for time series IDs for each `source_type`.
        default_values: Default values for each available feature.
        subsets: Available subsets for the database.
        matrix_feature_mappings: Which matrix ids maps to which matrices.
        source_types: Available source types for the database.
        aggregations: Available aggregations for the database.   
    """

    name: str
    bucket_url: str | None = None
    download_urls: dict[str, list[str | dict]] = {}
    fallback_download_urls: dict[str, list[str | dict]] = {}
    checksums: dict[str, str] = {}

    tszoo_root: str
    database_root: str
    configs_root: str
    benchmarks_root: str
    annotations_root: str

    id_names: dict = None
    default_values: dict = None
    subsets: list[str] = None
    matrix_feature_mappings: dict[str, str] = {}
    source_types: list[SourceType] = []
    aggregations: list[AgreggationType] = []

    def __init__(self):
        raise ValueError("To create dataset instance use class method 'get_dataset' instead.")

    @classmethod
    def get_dataset(cls, data_root: str, subset: str, source_type: SourceType | str, aggregation: AgreggationType | str, dataset_type: DatasetType | str, check_errors: bool = False, display_details: bool = False, backward_compatibility: bool = False, download_source: DownloadSource | str = "auto") -> CesnetDataset:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            subset: Specific subset of the dataset.
            source_type: The source type of the desired dataset.
            aggregation: The aggregation type for the selected source type.
            dataset_type: Type of a dataset you want to create. Can be [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) or [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset).
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`
            download_source: Where to download the dataset from when it is not present in `data_root` yet: `"auto"` tries every configured source in order (DataZoo bucket, then the mirrors) and moves on when one fails; `"datazoo"`, `"google_drive"` or `"zenodo"` uses only that source. See [`DownloadSource`][cesnet_tszoo.utils.enums.DownloadSource]. `Default: "auto"`

        Returns:
            [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) or [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset).
        """

        logger = logging.getLogger("wrapper_dataset")

        subset = subset.lower().strip() if subset is not None else subset
        source_type = SourceType(source_type)
        aggregation = AgreggationType(aggregation)
        dataset_type = DatasetType(dataset_type)
        download_source = DownloadSource(download_source)

        if cls.subsets is not None and subset not in cls.subsets:
            raise ValueError(f"Unsupported subset: {subset}. Supported are: {cls.subsets}")

        if source_type not in cls.source_types:
            raise ValueError(f"Unsupported source type: {source_type}. Supported are: {cls.source_types}")

        if aggregation not in cls.aggregations:
            raise ValueError(f"Unsupported aggregation type: {aggregation}. Supported are: {cls.aggregations}")

        # Dataset paths setup
        cls.tszoo_root = os.path.normpath(os.path.expanduser(os.path.join(data_root, "tszoo")))
        cls.database_root = os.path.join(cls.tszoo_root, "databases", cls.name)
        cls.configs_root = os.path.join(cls.tszoo_root, "configs")
        cls.benchmarks_root = os.path.join(cls.tszoo_root, "benchmarks")
        cls.annotations_root = os.path.join(cls.tszoo_root, "annotations")

        if backward_compatibility:
            name_aggregation = AgreggationType._to_str_without_number(aggregation)
        else:
            name_aggregation = aggregation.value

        if subset is None:
            dataset_name = f"{cls.name}-{source_type.value}-{name_aggregation}"
        else:
            dataset_name = f"{cls.name}-{subset}-{source_type.value}-{name_aggregation}"

        dataset_path = os.path.join(cls.database_root, f"{dataset_name}.h5")

        # Ensure necessary directories exist
        for directory in [cls.database_root, cls.configs_root, cls.annotations_root, cls.benchmarks_root]:
            if not os.path.exists(directory):
                logger.info("Creating directory: %s", directory)
                os.makedirs(directory)

        if not cls._is_downloaded(dataset_path):
            cls._download(dataset_name, dataset_path, download_source)

        dataset_metadata = DatasetMetadata(cls.name, dataset_type, dataset_path, cls.configs_root, cls.benchmarks_root, cls.annotations_root, subset, source_type, aggregation, cls.id_names[source_type], cls.default_values, cls.matrix_feature_mappings)

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
    def get_download_urls(cls, dataset_name: str, download_source: DownloadSource | str = DownloadSource.AUTO) -> list[str | dict]:
        """Returns all download sources for `dataset_name` in the order they are tried: `download_urls` mirrors first (plain URLs or multi-part sources, see `cesnet_tszoo.utils.download.parts_source`), then DataZoo `bucket_url`, then the `fallback_download_urls` mirrors. """

        urls = list(cls.download_urls.get(dataset_name, []))
        if cls.bucket_url is not None:
            urls.append(f"{cls.bucket_url}&file={dataset_name}.h5")
        urls.extend(cls.fallback_download_urls.get(dataset_name, []))

        download_source = DownloadSource(download_source)
        if download_source is not DownloadSource.AUTO:
            available = sorted({kind.value for kind in map(source_kind, urls) if kind is not None})
            urls = [url for url in urls if source_kind(url) is download_source]
            if len(urls) == 0:
                raise ValueError(f"{cls.name} has no '{download_source.value}' download source for {dataset_name}. "
                                 f"Available sources: {available} (or 'auto').")
        return urls

    @classmethod
    def _download(cls, dataset_name: str, dataset_path: str, download_source: DownloadSource | str = DownloadSource.AUTO) -> None:
        """Download the dataset file from the first working source. """

        logger = logging.getLogger("wrapper_dataset")

        logger.info("Downloading %s dataset.", dataset_name)
        used_url = download_from_sources(urls=cls.get_download_urls(dataset_name, download_source), file_path=dataset_path, expected_sha256=cls.checksums.get(dataset_name), silent=False)
        logger.info("Downloaded %s dataset from %s.", dataset_name, used_url)
