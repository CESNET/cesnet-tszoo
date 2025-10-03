from dataclasses import dataclass
import logging

from cesnet_tszoo.data_models.dataset_metadata import DatasetMetadata
from cesnet_tszoo.configs.config_editors.config_editor import ConfigEditor
from cesnet_tszoo.configs import SeriesBasedConfig


@dataclass
class SeriesBasedConfigEditor(ConfigEditor):
    """Used for choosing which values in config to modify."""

    default_config: SeriesBasedConfig

    def __post_init__(self):
        super().__post_init__()

        self.logger = logging.getLogger("series_based_config_editor")

    def _hard_modify(self, config: SeriesBasedConfig, dataset_metadata: DatasetMetadata):
        super()._hard_modify(config, dataset_metadata)

    def _soft_modify(self, config: SeriesBasedConfig, dataset_metadata: DatasetMetadata):
        super()._soft_modify(config, dataset_metadata)
