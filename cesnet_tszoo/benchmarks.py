from typing import Literal
import os
import logging

import pandas as pd

from cesnet_tszoo.files.utils import get_path_to_files_folder, get_benchmark_path_and_whether_it_is_built_in
from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.configs.time_based_config import TimeBasedConfig
from cesnet_tszoo.configs.series_based_config import SeriesBasedConfig

from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.datasets.datasets import CESNET_TimeSeries24, CesnetDatabase
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.utils.enums import AnnotationType, SourceType, AgreggationType
from cesnet_tszoo.utils.file_utils import yaml_load
from cesnet_tszoo.utils.utils import ExportBenchmark
from cesnet_tszoo.configs.config_loading import load_config


class Benchmark:
    """
    Used as wrapper for imported `dataset`, `config`, `annotations` and `related_results`.

    **Intended usage:**

    For time-based:

    When using [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset] (`dataset_type` = `DatasetType.TIME_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.cesnet_database.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_numpy].     

    When using [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] (`dataset_type` = `DatasetType.SERIES_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.cesnet_database.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_numpy].   

    When using [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset] (`dataset_type` = `DatasetType.DISJOINT_TIME_BASED`):

    1. Create an instance of the dataset with the desired data root by calling [`get_dataset`][cesnet_tszoo.datasets.cesnet_database.CesnetDatabase.get_dataset]. This will download the dataset if it has not been previously downloaded and return instance of dataset.
    2. Create an instance of [`DisjointTimeBasedConfig`][cesnet_tszoo.configs.combined_config.DisjointTimeBasedConfig] and set it using [`set_dataset_config_and_initialize`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.set_dataset_config_and_initialize]. 
       This initializes the dataset, including data splitting (train/validation/test), fitting transformers (if needed), selecting features, and more. This is cached for later use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_dataloader]/[`get_train_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_df]/[`get_train_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_train_numpy] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_dataloader]/[`get_val_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_df]/[`get_val_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_val_numpy].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_dataloader]/[`get_test_df`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_df]/[`get_test_numpy`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.get_test_numpy].      

    You can create custom time-based benchmarks with [`save_benchmark`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.save_benchmark], series-based benchmarks with [`save_benchmark`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.save_benchmark] or disjoint-time-based with [`save_benchmark`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset.save_benchmark].
    They will be saved to `"data_root"/tszoo/benchmarks/` directory, where `data_root` was set when you created instance of dataset.
    """

    def __init__(self, config: DatasetConfig, dataset: CesnetDataset, description: str = None):
        self.config = config
        self.dataset = dataset
        self.description = description
        self.related_results = None
        self.logger = logging.getLogger("benchmark")

    def get_config(self) -> SeriesBasedConfig | TimeBasedConfig:
        """Return config made for this benchmark. """

        return self.config

    def get_initialized_dataset(self, display_config_details: bool = True, check_errors: bool = False, workers: Literal["config"] | int = "config") -> TimeBasedCesnetDataset | SeriesBasedCesnetDataset:
        """
        Return dataset with intialized sets, transformers, fillers etc..

        This method uses following config attributes:

        | Dataset config                    | Description                                                                                    |
        | --------------------------------- | ---------------------------------------------------------------------------------------------- |
        | `init_workers`                    | Specifies the number of workers to use for initialization. Applied when `workers` = "config". |
        | `partial_fit_initialized_transformers` | Determines whether initialized transformers should be partially fitted on the training data.        |
        | `nan_threshold`                   | Filters out time series with missing values exceeding the specified threshold.                 |

        Parameters:
            display_config_details: Flag indicating whether to display the configuration values after initialization. `Default: True`   
            check_errors: Whether to validate if dataset is not corrupted. `Default: False`
            workers: The number of workers to use during initialization. `Default: "config"`        

        Returns:
            Return initialized dataset.
        """

        if check_errors:
            self.dataset.check_errors()

        self.dataset.set_dataset_config_and_initialize(self.config, display_config_details, workers)

        return self.dataset

    def get_dataset(self, check_errors: bool = False) -> TimeBasedCesnetDataset | SeriesBasedCesnetDataset:
        """Return dataset without initializing it.

        Parameters:
            check_errors: Whether to validate if dataset is not corrupted. `Default: False`

        Returns:
            Return dataset used for this benchmark.
        """

        if check_errors:
            self.dataset.check_errors()

        return self.dataset

    def get_annotations(self, on: AnnotationType | Literal["id_time", "ts_id", "both"]) -> pd.DataFrame:
        """ 
        Return the annotations as a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

        Parameters:
            on: Specifies which annotations to return. If set to `"both"`, annotations will be applied as if `id_time` and `ts_id` were both set.         

        Returns:
            A Pandas DataFrame containing the selected annotations.      
        """

        return self.dataset.get_annotations(on)

    def get_related_results(self) -> pd.DataFrame | None:
        """
        Return the related results as a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), if they exist. 

        Returns:
            A Pandas DataFrame containing related results or None if not related results exist. 
        """

        return self.related_results


def load_benchmark(identifier: str, data_root: str) -> Benchmark:
    """
    Load a benchmark using the identifier.

    First, it attempts to load the built-in benchmark, if no built-in benchmark with such an identifier exists, it attempts to load a custom benchmark from the `"data_root"/tszoo/benchmarks/` directory.

    Parameters:
        identifier: The name of the benchmark YAML file.
        data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `"data_root"/tszoo/databases/database_name/`.

    Returns:
        Return benchmark with `config`, `annotations`, `dataset` and `related_results`.
    """

    logger = logging.getLogger("benchmark")

    data_root = os.path.normpath(os.path.expanduser(data_root))

    # For anything else
    if isinstance(identifier, str):
        _, is_built_in = get_benchmark_path_and_whether_it_is_built_in(identifier, data_root, logger)

        if is_built_in:
            logger.info("Built-in benchmark found: %s. Loading it.", identifier)
            return _get_built_in_benchmark(identifier, data_root)
        else:
            logger.info("Custom benchmark found: %s. Loading it.", identifier)
            return _get_custom_benchmark(identifier, data_root)

    else:
        logger.error("Invalid identifier.")
        raise ValueError("Invalid identifier.")


def _get_dataset(data_root: str, export_benchmark: ExportBenchmark) -> TimeBasedCesnetDataset | SeriesBasedCesnetDataset:
    """Returns `dataset` based on `export_benchmark`. """

    if export_benchmark.database_name == CESNET_TimeSeries24.name:
        dataset = CESNET_TimeSeries24.get_dataset(data_root, export_benchmark.source_type, export_benchmark.aggregation, export_benchmark.dataset_type, False, False)
    else:
        raise ValueError("Invalid database name.")

    return dataset


def _get_built_in_benchmark(identifier: str, data_root: str) -> Benchmark:
    """Returns built-in benchmark. Looks for benchmark in built-in folder in the package."""

    logger = logging.getLogger("benchmark")

    path_for_related_results = os.path.join(get_path_to_files_folder(), "related_results")
    path_for_built_in_benchmarks = os.path.join(get_path_to_files_folder(), "benchmark_files")

    # Load the benchmark file
    benchmark_file_path = os.path.join(path_for_built_in_benchmarks, f"{identifier}.yaml")
    logger.debug("Loading benchmark from '%s'.", benchmark_file_path)
    export_benchmark = ExportBenchmark.from_dict(yaml_load(benchmark_file_path))

    config_root = CesnetDatabase.get_expected_paths(data_root, export_benchmark.database_name)["configs_root"]

    config = load_config(export_benchmark.config_identifier, config_root, export_benchmark.database_name, SourceType(export_benchmark.source_type), AgreggationType(export_benchmark.aggregation), logger)

    if export_benchmark.dataset_type is None:
        export_benchmark.dataset_type = config.dataset_type

    # Prepare the dataset
    dataset = _get_dataset(data_root, export_benchmark)

    # Check and load annotations if available
    if export_benchmark.annotations_ts_identifier is not None:
        dataset.import_annotations(export_benchmark.annotations_ts_identifier, enforce_ids=False)
    else:
        logger.info("No %s annotations found.", AnnotationType.TS_ID)

    if export_benchmark.annotations_time_identifier is not None:
        dataset.import_annotations(export_benchmark.annotations_time_identifier, enforce_ids=False)
    else:
        logger.info("No %s annotations found.", AnnotationType.ID_TIME)

    if export_benchmark.annotations_both_identifier is not None:
        dataset.import_annotations(export_benchmark.annotations_both_identifier, enforce_ids=False)
    else:
        logger.info("No %s annotations found.", AnnotationType.BOTH)

    logger.debug("Creating benchmark with description '%s'.", export_benchmark.description)
    result_benchmark = Benchmark(config, dataset, export_benchmark.description)

    # Load related results if available
    if export_benchmark.related_results_identifier is not None:
        related_results_file_path = os.path.join(path_for_related_results, f"{export_benchmark.related_results_identifier}.csv")
        logger.debug("Loading related results from '%s'.", related_results_file_path)
        result_benchmark.related_results = pd.read_csv(related_results_file_path)
        logger.info("Related results found and loaded.")
    else:
        logger.info("No related results found for benchmark '%s'.", identifier)

    logger.info("Built-in benchmark '%s' successfully prepared and ready for use.", identifier)

    return result_benchmark


def _get_custom_benchmark(identifier: str, data_root: str) -> Benchmark:
    """Returns custom benchmark. Looks for benchmark in `data_root`."""

    logger = logging.getLogger("benchmark")

    benchmark_file_path = os.path.join(data_root, "tszoo", "benchmarks", f"{identifier}.yaml")
    logger.debug("Looking for benchmark configuration file at '%s'.", benchmark_file_path)

    if not os.path.exists(benchmark_file_path):
        logger.error("Benchmark '%s' not found at expected path '%s'.", identifier, benchmark_file_path)
        raise ValueError(f"Benchmark {identifier} not found on path {benchmark_file_path}")

    # Load the benchmark file
    export_benchmark = ExportBenchmark.from_dict(yaml_load(benchmark_file_path))
    logger.info("Loaded benchmark '%s' with description: '%s'.", identifier, export_benchmark.description)

    config_root = CesnetDatabase.get_expected_paths(data_root, export_benchmark.database_name)["configs_root"]

    # Load config
    config = load_config(export_benchmark.config_identifier, config_root, export_benchmark.database_name, SourceType(export_benchmark.source_type), AgreggationType(export_benchmark.aggregation), logger)

    if export_benchmark.dataset_type is None:
        export_benchmark.dataset_type = config.dataset_type

    # Prepare the dataset
    dataset = _get_dataset(data_root, export_benchmark)

    # Load annotations if available
    if export_benchmark.annotations_ts_identifier is not None:
        dataset.import_annotations(export_benchmark.annotations_ts_identifier, enforce_ids=False)
    else:
        logger.info("No %s annotations found.", AnnotationType.TS_ID)

    if export_benchmark.annotations_time_identifier is not None:
        dataset.import_annotations(export_benchmark.annotations_time_identifier, enforce_ids=False)
    else:
        logger.info("No %s annotations found.", AnnotationType.ID_TIME)

    if export_benchmark.annotations_both_identifier is not None:
        dataset.import_annotations(export_benchmark.annotations_both_identifier, enforce_ids=False)
    else:
        logger.info("No %s annotations found.", AnnotationType.BOTH)

    # Since the benchmark is custom, related results are None
    logger.info("As benchmark '%s' is custom, related results cant be loaded.", identifier)

    logger.debug("Creating benchmark with description '%s'.", export_benchmark.description)
    result_benchmark = Benchmark(config, dataset, export_benchmark.description)

    logger.info("Custom benchmark '%s' successfully prepared and ready for use.", identifier)

    return result_benchmark
