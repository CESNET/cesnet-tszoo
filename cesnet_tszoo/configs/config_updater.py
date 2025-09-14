import logging

from packaging.version import Version
from copy import deepcopy

from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.utils.filler import get_filler_factory
from cesnet_tszoo.utils.anomaly_handler import get_anomaly_handler_factory
from cesnet_tszoo.utils.enums import DatasetType, TransformerType, ScalerType
import cesnet_tszoo.version as version


class ConfigUpdater:

    def __init__(self, config: DatasetConfig):
        self.logger = logging.getLogger("config_updater")
        self.config_to_update = deepcopy(config)
        self.updated_config = None

    def try_get_updated_config(self) -> DatasetConfig:
        """Tries to update config to match newer version of library. """
        if self.updated_config is not None:
            return self.updated_config

        self.logger.debug("Trying to update config if necessary.")
        self.__try_set_default_config_version()

        self.__validate_whether_config_update_possible_()

        if self.config_to_update.version == version.current_version:
            self.updated_config = self.config_to_update
            return self.updated_config

        if Version(self.config_to_update.version) < Version(version.VERSION_0_1_3):
            self.logger.warning("Config version is lower than '%s', updating config to match it.", version.VERSION_0_1_3)

            self.__scaler_to_transformer_version_update()
            self.__update_to_new_config_type_support()
            self.__add_anomaly_handler()

            self.config_to_update.export_update_needed = True
            self.config_to_update.version = version.VERSION_0_1_3
            self.logger.debug("Updating config version to %s used cesnet-tszoo package version.", version.VERSION_0_1_3)

        self.__default_version_update(version.VERSION_2_0_0)

        if Version(self.config_to_update.version) < Version(version.VERSION_2_0_1):
            self.logger.warning("Config version is lower than '%s', updating config to match it.", version.VERSION_2_0_1)

            self.__filler_refactoring()
            self.__anomaly_handler_refactoring()

            self.config_to_update.export_update_needed = True
            self.config_to_update.version = version.VERSION_2_0_1
            self.logger.debug("Updating config version to %s used cesnet-tszoo package version.", version.VERSION_2_0_1)

        self.logger.debug("Updating config version to %s used cesnet-tszoo package version.", version.current_version)
        self.config_to_update.version = version.current_version

        self.updated_config = self.config_to_update

        return self.updated_config

    def __filler_refactoring(self):
        self.logger.debug("Updating attributes for filler refactoring.")

        self.config_to_update.filler_factory = get_filler_factory(getattr(self.config_to_update, "fill_missing_with"))

        delattr(self.config_to_update, "fill_missing_with")
        delattr(self.config_to_update, "is_filler_custom")
        delattr(self.config_to_update, "fill_missing_with_display")
        delattr(self.config_to_update, "used_fillers")

    def __anomaly_handler_refactoring(self):
        self.logger.debug("Updating attributes for anomaly handler refactoring.")

        self.config_to_update.anomaly_handler_factory = get_anomaly_handler_factory(getattr(self.config_to_update, "handle_anomalies_with"))

        delattr(self.config_to_update, "handle_anomalies_with")
        delattr(self.config_to_update, "is_anomaly_handler_custom")
        delattr(self.config_to_update, "handle_anomalies_with_display")
        delattr(self.config_to_update, "used_anomaly_handlers")

    def __default_version_update(self, update_version: str):
        if Version(self.config_to_update.version) < Version(update_version):
            self.logger.warning("Config version is lower than '%s', updating config to match it.", update_version)
            self.config_to_update.export_update_needed = True
            self.config_to_update.version = update_version
            self.logger.debug("Updating config version to %s used cesnet-tszoo package version.", update_version)

    def __try_set_default_config_version(self):
        if not hasattr(self.config_to_update, "version"):
            self.logger.debug("Config attribute 'version' is missing in this instance. Default version '%s' will be temporarily set.", version.DEFAULT_VERSION)
            self.config_to_update.version = version.DEFAULT_VERSION

    def __validate_whether_config_update_possible_(self):
        if Version(self.config_to_update.version) < Version(version.current_version):
            self.logger.warning("Imported config was made for cesnet-tszoo package of version '%s', but current used cesnet-tszoo package version is '%s'!", self.config_to_update.version, version.current_version)
            self.logger.warning("Will try to update the config. It is recommended to recreate this config or at least export this config alone or through benchmark to create updated config file.")
        elif Version(self.config_to_update.version) > Version(version.current_version):
            self.logger.error("Imported config was made for cesnet-tszoo package of version '%s', but current used cesnet-tszoo package version is '%s'!", self.config_to_update.version, version.current_version)
            self.logger.error("Update cesnet-tszoo package to use this config.")
            raise ValueError(f"Imported config was made for cesnet-tszoo package of version '{self.config_to_update.version}', but current used cesnet-tszoo package version is '{version.current_version}'!")

    def __scaler_to_transformer_version_update(self):
        self.logger.debug("Updating attributes from scaler variant to transformer variant.")

        self.config_to_update.transform_with = getattr(self.config_to_update, "scale_with")
        delattr(self.config_to_update, "scale_with")

        if self.config_to_update.transform_with is not None and isinstance(self.config_to_update.transform_with, ScalerType):
            self.config_to_update.transform_with = TransformerType(self.config_to_update.transform_with.value)

        self.config_to_update.partial_fit_initialized_transformers = getattr(self.config_to_update, "partial_fit_initialized_scalers")
        delattr(self.config_to_update, "partial_fit_initialized_scalers")

        self.config_to_update.create_transformer_per_time_series = getattr(self.config_to_update, "create_scaler_per_time_series")
        delattr(self.config_to_update, "create_scaler_per_time_series")

        self.config_to_update.transform_with_display = getattr(self.config_to_update, "scale_with_display")
        delattr(self.config_to_update, "scale_with_display")

        self.config_to_update.is_transformer_custom = getattr(self.config_to_update, "is_scaler_custom")
        delattr(self.config_to_update, "is_scaler_custom")

        self.config_to_update.transformers = getattr(self.config_to_update, "scalers")
        delattr(self.config_to_update, "scalers")

        self.config_to_update.are_transformers_premade = getattr(self.config_to_update, "are_scalers_premade")
        delattr(self.config_to_update, "are_scalers_premade")

    def __update_to_new_config_type_support(self):
        self.logger.debug("Updating config to work with new config type.")

        is_series_based = getattr(self.config_to_update, "is_series_based")
        delattr(self.config_to_update, "is_series_based")
        delattr(self.config_to_update, "has_train")
        delattr(self.config_to_update, "has_val")
        delattr(self.config_to_update, "has_test")
        delattr(self.config_to_update, "has_all")

        if is_series_based:
            self.logger.debug("Updating config as SeriesBasedConfig.")
            self.config_to_update.dataset_type = DatasetType.SERIES_BASED

            self.config_to_update.uses_all_ts = True
            delattr(self.config_to_update, "sliding_window_size")
            delattr(self.config_to_update, "sliding_window_prediction_size")
            delattr(self.config_to_update, "sliding_window_step")
            delattr(self.config_to_update, "set_shared_size")

        elif self.config_to_update.test_ts_ids is None:
            self.logger.debug("Updating config as TimeBasedConfig.")
            self.config_to_update.dataset_type = DatasetType.TIME_BASED
            self.config_to_update.uses_all_time_period = True
            delattr(self.config_to_update, "used_test_other_workers")
            delattr(self.config_to_update, "test_ts_row_ranges")
            delattr(self.config_to_update, "other_test_fillers")
            delattr(self.config_to_update, "has_ts_ids")
            delattr(self.config_to_update, "has_test_ts_ids")
            delattr(self.config_to_update, "used_singular_test_other_time_series")

        else:
            raise ValueError("Cannot update config, because it uses test_ts_ids which cannot be easily converted to newer format.")

    def __add_anomaly_handler(self):
        self.logger.debug("Adding anomaly handler attributes.")

        self.config_to_update.handle_anomalies_with = None
        self.config_to_update.handle_anomalies_with_display = None
        self.config_to_update.is_anomaly_handler_custom = False
        self.config_to_update.anomaly_handlers = None
        self.config_to_update.used_anomaly_handlers = None
