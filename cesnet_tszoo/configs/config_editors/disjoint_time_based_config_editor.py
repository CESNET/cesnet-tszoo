from dataclasses import dataclass, field
from typing import Literal
import logging

from cesnet_tszoo.data_models.dataset_metadata import DatasetMetadata
from cesnet_tszoo.configs.config_editors.config_editor import ConfigEditor
from cesnet_tszoo.configs import DisjointTimeBasedConfig


@dataclass
class DisjointTimeBasedConfigEditor(ConfigEditor):
    """Used for choosing which values in config to modify."""

    default_config: DisjointTimeBasedConfig
    sliding_window_size: int | None | Literal["config"]
    sliding_window_prediction_size: int | None | Literal["config"]
    sliding_window_step: int | Literal["config"]
    set_shared_size: float | int | Literal["config"]

    all_batch_size: int = field(default=1, init=False)
    all_workers: int = field(default=1, init=False)

    def __post_init__(self):
        if self.sliding_window_size == "config":
            self.sliding_window_size = self.default_config.sliding_window_size
        if self.sliding_window_prediction_size == "config":
            self.sliding_window_prediction_size = self.default_config.sliding_window_prediction_size
        if self.sliding_window_step == "config":
            self.sliding_window_step = self.default_config.sliding_window_step
        if self.set_shared_size == "config":
            self.set_shared_size = self.default_config.set_shared_size
        else:
            self.requires_init = True

        super().__post_init__()

        self.logger = logging.getLogger("disjoint_time_based_config_editor")

    def _hard_modify(self, config: DisjointTimeBasedConfig, dataset_metadata: DatasetMetadata):
        super()._hard_modify(config, dataset_metadata)

    def _soft_modify(self, config: DisjointTimeBasedConfig, dataset_metadata: DatasetMetadata):
        super()._soft_modify(config, dataset_metadata)
        config._update_sliding_window(self.sliding_window_size, self.sliding_window_prediction_size, self.sliding_window_step, self.set_shared_size, dataset_metadata.time_indices)
