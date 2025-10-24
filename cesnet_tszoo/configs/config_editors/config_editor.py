from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal
from numbers import Number
import logging

import numpy as np

from cesnet_tszoo.data_models.dataset_metadata import DatasetMetadata
from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.utils.enums import FillerType, TransformerType, AnomalyHandlerType
from cesnet_tszoo.utils.transformer import Transformer
import cesnet_tszoo.utils.filler.factory as filler_factories
import cesnet_tszoo.utils.transformer.factory as transformer_factories
import cesnet_tszoo.utils.anomaly_handler.factory as anomaly_handler_factories


@dataclass
class ConfigEditor(ABC):
    """Used for choosing which values in config to modify."""

    default_config: DatasetConfig
    default_values: list[Number] | dict[str, Number] | Number | Literal["default"] | None | Literal["config"]
    train_batch_size: int | Literal["config"]
    val_batch_size: int | Literal["config"]
    test_batch_size: int | Literal["config"]
    all_batch_size: int | Literal["config"]
    preprocess_order: list[str] | Literal["config"]
    fill_missing_with: type | FillerType | Literal["mean_filler", "forward_filler", "linear_interpolation_filler"] | None | Literal["config"]
    transform_with: type | list[Transformer] | np.ndarray[Transformer] | TransformerType | Transformer | Literal["min_max_scaler", "standard_scaler", "max_abs_scaler", "log_transformer", "robust_scaler", "power_transformer", "quantile_transformer", "l2_normalizer"] | None | Literal["config"]
    handle_anomalies_with: type | AnomalyHandlerType | Literal["z-score", "interquartile_range"] | None | Literal["config"]
    create_transformer_per_time_series: bool | Literal["config"]
    partial_fit_initialized_transformers: bool | Literal["config"]
    train_workers: int | Literal["config"]
    val_workers: int | Literal["config"]
    test_workers: int | Literal["config"]
    all_workers: int | Literal["config"]
    init_workers: int | Literal["config"]
    requires_init: bool = field(default=False, init=False)

    def __post_init__(self):
        self.logger = logging.getLogger("config_editor")

        if self.default_values == "config":
            self.default_values = self.default_config.default_values
        else:
            self.requires_init = True

        if self.preprocess_order == "config":
            self.preprocess_order = self.default_config.preprocess_order
        else:
            self.requires_init = True

        if self.train_batch_size == "config":
            self.train_batch_size = self.default_config.train_batch_size
        if self.val_batch_size == "config":
            self.val_batch_size = self.default_config.val_batch_size
        if self.test_batch_size == "config":
            self.test_batch_size = self.default_config.test_batch_size
        if self.all_batch_size == "config":
            self.all_batch_size = self.default_config.all_batch_size

        if self.fill_missing_with == "config":
            self.fill_missing_with = self.default_config.filler_factory.filler_type
        else:
            self.requires_init = True

        if self.create_transformer_per_time_series == "config":
            self.create_transformer_per_time_series = self.default_config.create_transformer_per_time_series
        else:
            self.requires_init = True

        if self.partial_fit_initialized_transformers == "config":
            self.partial_fit_initialized_transformers = self.default_config.partial_fit_initialized_transformers
        else:
            self.requires_init = True

        if self.transform_with == "config":
            if self.default_config.transformer_factory.has_already_initialized:
                self.transform_with = self.default_config.transformer_factory.initialized_transformers
            else:
                self.transform_with = self.default_config.transformer_factory.transformer_type
        else:
            self.requires_init = True

        if self.handle_anomalies_with == "config":
            self.handle_anomalies_with = self.default_config.anomaly_handler_factory.anomaly_handler_type
        else:
            self.requires_init = True

        if self.train_workers == "config":
            self.train_workers = self.default_config.train_workers
        if self.val_workers == "config":
            self.val_workers = self.default_config.val_workers
        if self.test_workers == "config":
            self.test_workers = self.default_config.test_workers
        if self.all_workers == "config":
            self.all_workers = self.default_config.all_workers
        if self.init_workers == "config":
            self.init_workers = self.default_config.init_workers

    def modify_dataset_config(self, dataset_config: DatasetConfig, metadata: DatasetMetadata):
        """Modifies dataset config based on passed values in constructor. Used by CesnetDataset classes when editing config values. """

        if self.requires_init:
            self._soft_modify(dataset_config, metadata)
            self._hard_modify(dataset_config, metadata)
            dataset_config._validate_construction()
        else:
            self._soft_modify(dataset_config, metadata)

    @abstractmethod
    def _hard_modify(self, config: DatasetConfig, dataset_metadata: DatasetMetadata):
        config.default_values = self.default_values
        config.preprocess_order = self.preprocess_order
        config.partial_fit_initialized_transformers = self.partial_fit_initialized_transformers
        config.create_transformer_per_time_series = self.create_transformer_per_time_series
        config.filler_factory = filler_factories.get_filler_factory(self.fill_missing_with)
        config.transformer_factory = transformer_factories.get_transformer_factory(self.transform_with, self.create_transformer_per_time_series, self.partial_fit_initialized_transformers)
        config.anomaly_handler_factory = anomaly_handler_factories.get_anomaly_handler_factory(self.handle_anomalies_with)

    @abstractmethod
    def _soft_modify(self, config: DatasetConfig, dataset_metadata: DatasetMetadata):
        config._update_batch_sizes(self.train_batch_size, self.val_batch_size, self.test_batch_size, self.all_batch_size)
        config._update_workers(self.train_workers, self.val_workers, self.test_workers, self.all_workers, self.init_workers)
