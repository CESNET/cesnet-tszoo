from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.utils.enums import SplitType
from cesnet_tszoo.data_models.dataset_metadata import DatasetMetadata


class DatasetInitConfig(ABC):
    """Base class for dataset init configs that are used to pass values to init datasets. """

    def __init__(self, config: DatasetConfig, limit_init_to_set: Optional[SplitType], dataset_metadata: DatasetMetadata):
        self.ts_id_name = config.ts_id_name

        self.all_features_to_take_table = config.features_to_take.copy()
        for i, feature in enumerate(self.all_features_to_take_table):
            for feature_id in dataset_metadata.matrix_feature_mappings:
                if feature == dataset_metadata.matrix_feature_mappings[feature_id]:
                    self.all_features_to_take_table[i] = feature_id
                    break

        self.non_matrix_features_to_take = [feature for feature in config.features_to_take if feature not in dataset_metadata.matrix_feature_mappings.values()]
        self.non_matrix_feature_indices = [index for index, feature in enumerate(config.features_to_take) if feature in self.non_matrix_features_to_take]

        self.matrix_feature_indices = [index for index, feature in enumerate(config.features_to_take) if feature not in self.non_matrix_features_to_take]
        self.matrix_feature_ids_to_take = [feature_id for feature_id in dataset_metadata.matrix_feature_mappings if dataset_metadata.matrix_feature_mappings[feature_id] in config.features_to_take]
        self.matrix_features_to_take = [dataset_metadata.matrix_feature_mappings[feature_id] for feature_id in dataset_metadata.matrix_feature_mappings if dataset_metadata.matrix_feature_mappings[feature_id] in config.features_to_take]

        self.non_id_scalar_features_count = len(config.indices_of_features_to_take_no_ids) - len(self.matrix_features_to_take)

        self.time_period = None
        self.nan_threshold = config.nan_threshold

        if limit_init_to_set == SplitType.TRAIN:
            self._init_train(config)
        elif limit_init_to_set == SplitType.VAL:
            self._init_val(config)
        elif limit_init_to_set == SplitType.TEST:
            self._init_test(config)
        elif limit_init_to_set == SplitType.ALL:
            self._init_all(config)
        else:
            self._init_train(config)
            self._init_val(config)
            self._init_test(config)
            self._init_all(config)

    @abstractmethod
    def _init_train(self, config: DatasetConfig):
        """Initializes from train data of config """
        ...

    @abstractmethod
    def _init_val(self, config: DatasetConfig):
        """Initializes from val data of config """
        ...

    @abstractmethod
    def _init_test(self, config: DatasetConfig):
        """Initializes from test data of config """
        ...

    @abstractmethod
    def _init_all(self, config: DatasetConfig):
        """Initializes from all data of config """
        ...
