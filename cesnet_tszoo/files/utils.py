import os
import pandas as pd
import logging

from cesnet_tszoo.utils.utils import is_config_used_for_dataset
from cesnet_tszoo.utils.file_utils import pickle_load
from cesnet_tszoo.utils.enums import SourceType, AgreggationType


def get_path_to_files_folder() -> str:
    """Return path to folder with benchmark, configs, annotations and related_results that came with this package. """

    return os.path.dirname(__file__)


def get_benchmark_path_and_whether_it_is_built_in(identifier: str, data_root: str, logger: logging.Logger) -> tuple[str, bool]:
    """Returns path for benchmark if it exists and whether it is built-in."""

    logger.debug("Checking for built-in benchmark.")
    path_for_built_in_benchmark = os.path.join(get_path_to_files_folder(), "benchmark_files", f"{identifier}.yaml")
    is_built_in = exists_built_in_benchmark(identifier)

    if not is_built_in:
        logger.warning("Built-in benchmark %s not found.", identifier)

    if is_built_in:
        return path_for_built_in_benchmark, True

    logger.debug("Checking for custom benchmark.")
    path_for_custom_benchmark = os.path.join(data_root, "tszoo", "benchmarks", f"{identifier}.yaml")
    is_custom = os.path.exists(path_for_custom_benchmark)

    if not is_custom:
        logger.warning("No benchmark with identifier %s found at expected path: %s.", identifier, path_for_custom_benchmark)

    if is_custom:
        return path_for_custom_benchmark, False

    logger.error("No benchmark with identifier %s found.", identifier)
    raise FileNotFoundError(f"No benchmark with identifer {identifier} was found.")


def get_config_path_and_whether_it_is_built_in(identifier: str, custom_configs_root: str, dataset_database_name: str, dataset_source_type: SourceType, dataset_aggregation: AgreggationType, logger: logging.Logger) -> tuple[str, bool]:
    """Returns path for config if it exists and whether it is built-in."""

    logger.debug("Checking for built-in config.")
    path_for_built_in_config = os.path.join(get_path_to_files_folder(), "config_files", f"{identifier}.pickle")
    is_built_in = exists_built_in_config(identifier)

    if not is_built_in:
        logger.warning("Built-in config %s not found.", identifier)

    if is_built_in:
        config = pickle_load(path_for_built_in_config)
        if not is_config_used_for_dataset(config, dataset_database_name, dataset_source_type, dataset_aggregation):
            logger.error("Config with identifier %s can't be used for this dataset. Config is used for dataset of database: %s, source_type: %s, aggregation: %s", identifier, config.database_name, config.source_type, config.aggregation)
            raise ValueError(f"Config with identifier {identifier} can't be used for this dataset.")
        return path_for_built_in_config, True

    logger.debug("Checking for custom config.")
    path_for_custom_config = os.path.join(custom_configs_root, f"{identifier}.pickle")
    is_custom = os.path.exists(path_for_custom_config)

    if not is_custom:
        logger.warning("No config with identifier %s found at expected path: %s.", identifier, path_for_custom_config)

    if is_custom:
        logger.debug("Loading config file from '%s'.", path_for_custom_config)
        config = pickle_load(path_for_custom_config)
        if not is_config_used_for_dataset(config, dataset_database_name, dataset_source_type, dataset_aggregation):
            logger.error("Config with identifier %s can't be used for this dataset. Config is used for dataset of database: %s, source_type: %s, aggregation: %s", identifier, config.database_name, config.source_type, config.aggregation)
            raise ValueError(f"Config with identifier {identifier} can't be used for this dataset.")
        return path_for_custom_config, False

    logger.error("No config with identifier %s found.", identifier)
    raise FileNotFoundError(f"No config with identifer {identifier} was found.")


def get_annotations_path_and_whether_it_is_built_in(identifier: str, custom_annotations_root: str, logger: logging.Logger) -> tuple[str, bool]:
    """Returns path for annotations if it exists and whether it is built-in."""

    logger.debug("Checking for built-in annotations.")
    path_for_built_in_annotations = os.path.join(get_path_to_files_folder(), "annotation_files", f"{identifier}.csv")
    is_built_in = exists_built_in_annotations(identifier)

    if not is_built_in:
        logger.warning("Built-in annotations %s not found.", identifier)

    if is_built_in:
        return path_for_built_in_annotations, True

    logger.debug("Checking for custom annotations.")
    path_for_custom_annotations = os.path.join(custom_annotations_root, f"{identifier}.csv")
    is_custom = os.path.exists(path_for_custom_annotations)

    if not is_custom:
        logger.warning("No annotations with identifier %s found at expected path: %s.", identifier, path_for_custom_annotations)

    if is_custom:
        return path_for_custom_annotations, False

    logger.error("No annotations with identifier %s found.", identifier)
    raise FileNotFoundError(f"No annotations with identifer {identifier} was found.")


def exists_built_in_benchmark(identifier: str) -> bool:
    """Whether benchmark with identifier is already built-in."""
    path_for_built_in_benchmark = os.path.join(get_path_to_files_folder(), "benchmark_files", f"{identifier}.yaml")
    return os.path.exists(path_for_built_in_benchmark)


def exists_built_in_config(identifier: str) -> bool:
    """Whether config with identifier is already built-in."""
    path_for_built_in_config = os.path.join(get_path_to_files_folder(), "config_files", f"{identifier}.pickle")
    return os.path.exists(path_for_built_in_config)


def exists_built_in_annotations(identifier: str) -> bool:
    """Whether annotations with identifier is already built-in."""
    return is_annotations_built_in_for_dataset(identifier)


def is_annotations_built_in_for_dataset(identifier: str) -> bool:
    """Checks whether built-in annotations with identifier exist. """
    annotations_metadata_file_path = os.path.join(get_path_to_files_folder(), "annotations_metadata.csv")

    return _is_built_in_for_dataset(annotations_metadata_file_path, identifier)


def _is_built_in_for_dataset(metadata_file_path: str, identifier: str) -> bool:
    metadata_df = pd.read_csv(metadata_file_path)

    metadata_df = metadata_df.query('`identifier` == @identifier')
    if len(metadata_df) == 0:
        return False

    return True
