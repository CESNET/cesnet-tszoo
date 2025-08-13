from logging import Logger

from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.files.utils import get_config_path_and_whether_it_is_built_in
from cesnet_tszoo.utils.file_utils import pickle_load
from cesnet_tszoo.utils.enums import SourceType, AgreggationType
from cesnet_tszoo.configs.config_updater import ConfigUpdater


def load_config(identifier: str, config_root: str, database_name: str, source_type: str, aggregation: str, logger: Logger) -> DatasetConfig:
    config_file_path, is_built_in = get_config_path_and_whether_it_is_built_in(identifier, config_root, database_name, SourceType(source_type), AgreggationType(aggregation), logger)

    if is_built_in:
        logger.info("Built-in config found: %s. Loading it.", identifier)
        config = pickle_load(config_file_path)
    else:
        logger.info("Custom config found: %s. Loading it.", identifier)
        config = pickle_load(config_file_path)

    logger.info("Successfully loaded config from %s", config_file_path)

    config.import_identifier = identifier

    config_updater = ConfigUpdater(config)

    config = config_updater.try_get_updated_config()

    return config
