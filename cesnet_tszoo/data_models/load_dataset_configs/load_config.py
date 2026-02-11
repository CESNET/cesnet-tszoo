from abc import ABC, abstractmethod
from typing import Optional

from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.data_models.preprocess_note import PreprocessNote
from cesnet_tszoo.utils.enums import SplitType
from cesnet_tszoo.data_models.dataset_metadata import DatasetMetadata


class LoadConfig(ABC):
    """Base class for dataset configs that are used to pass values to load datasets. """

    def __init__(self, config: DatasetConfig, limit_init_to_set: Optional[SplitType], dataset_metadata: DatasetMetadata):
        self.ts_id_name = config.ts_id_name
        self.ts_row_ranges = None
        self.time_period = None
        self.include_time = config.include_time
        self.include_ts_id = config.include_ts_id
        self.time_format = config.time_format
        self.preprocess_order: list[PreprocessNote] = None

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

        self.return_dtype = config._get_dataloader_return_dtype(dataset_metadata)
        self.return_dtype_fill_values = config._get_dataloader_return_dtype_fill_values(dataset_metadata)

        if limit_init_to_set == SplitType.TRAIN:
            self._init_train(config)
        elif limit_init_to_set == SplitType.VAL:
            self._init_val(config)
        elif limit_init_to_set == SplitType.TEST:
            self._init_test(config)
        elif limit_init_to_set == SplitType.ALL:
            self._init_all(config)
        else:
            raise NotImplementedError()

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

    @abstractmethod
    def create_split_copy(self, split_range: slice):
        """Creates copy with splitted values. """
        ...
