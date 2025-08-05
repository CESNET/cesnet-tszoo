from typing import Literal
import os
import logging

import pandas as pd

from cesnet_tszoo.files.utils import get_path_to_files_folder, get_benchmark_path_and_whether_it_is_built_in, get_config_path_and_whether_it_is_built_in
from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.configs.time_based_config import TimeBasedConfig
from cesnet_tszoo.configs.series_based_config import SeriesBasedConfig

from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.datasets.datasets import CESNET_TimeSeries24
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.utils.enums import AnnotationType, SourceType, AgreggationType
from cesnet_tszoo.utils.file_utils import yaml_load, pickle_load
from cesnet_tszoo.utils.utils import ExportBenchmark


class Benchmark:
    """
    Used as wrapper for imported `dataset`, `config`, `annotations` and `related_results`.

    **Intended usage:**

    For time-based:

    1. Call [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark] with the desired benchmark identifier. You can use your own saved benchmark or you can use already built-in one. This will download the dataset and annotations (if available) if they have not been previously downloaded.
    2. Retrieve the initialized dataset using [`get_initialized_dataset`][cesnet_tszoo.benchmarks.Benchmark.get_initialized_dataset]. This will provide a dataset that is ready to use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_dataloader] or [`get_train_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_train_df] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_dataloader] or [`get_val_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_val_df].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_dataloader] or [`get_test_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_df]. 
    6. (Optional) Evaluate the model on [`get_test_other_dataloader`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_dataloader] or [`get_test_other_df`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.get_test_other_df]. 

    For series-based: 

    1. Call [`load_benchmark`][cesnet_tszoo.benchmarks.load_benchmark] with the desired benchmark. You can use your own saved benchmark or you can use already built-in one. This will download the dataset and annotations (if available) if they have not been previously downloaded.
    2. Retrieve the initialized dataset using [`get_initialized_dataset`][cesnet_tszoo.benchmarks.Benchmark.get_initialized_dataset]. This will provide a dataset that is ready to use.
    3. Use [`get_train_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_dataloader] or [`get_train_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_train_df] to get training data for chosen model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_dataloader] or [`get_val_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_val_df].
    5. Evaluate the model on [`get_test_dataloader`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_dataloader] or [`get_test_df`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.get_test_df].     

    You can create custom time-based benchmarks with [`save_benchmark`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset.save_benchmark] or series-based benchmarks with [`save_benchmark`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset.save_benchmark].
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
        dataset = CESNET_TimeSeries24.get_dataset(data_root, export_benchmark.source_type, export_benchmark.aggregation, export_benchmark.is_series_based, False, False)
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

    # Prepare the dataset
    dataset = _get_dataset(data_root, export_benchmark)

    # Load config
    config_file_path, _ = get_config_path_and_whether_it_is_built_in(export_benchmark.config_identifier, dataset.configs_root, export_benchmark.database_name, SourceType(export_benchmark.source_type), AgreggationType(export_benchmark.aggregation), logger)
    logger.debug("Loading config file from '%s'.", config_file_path)
    config = pickle_load(config_file_path)
    config.import_identifier = export_benchmark.config_identifier
    config._try_backward_support_update()

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

    # Prepare the dataset
    dataset = _get_dataset(data_root, export_benchmark)

    # Load config
    config_file_path, is_built_in = get_config_path_and_whether_it_is_built_in(export_benchmark.config_identifier, dataset.configs_root, export_benchmark.database_name, SourceType(export_benchmark.source_type), AgreggationType(export_benchmark.aggregation), logger)

    if is_built_in:
        logger.info("Built-in config found: %s. Loading it.", identifier)
        config = pickle_load(config_file_path)
    else:
        logger.info("Custom config found: %s. Loading it.", identifier)
        config = pickle_load(config_file_path)

    config.import_identifier = export_benchmark.config_identifier
    config._try_backward_support_update()

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
