from abc import ABC, abstractmethod
from typing import Optional

from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.data_models.preprocess_note import PreprocessNote
from cesnet_tszoo.utils.enums import SplitType


class LoadConfig(ABC):
    """Base class for dataset configs that are used to pass values to load datasets. """

    def __init__(self, config: DatasetConfig, limit_init_to_set: Optional[SplitType]):
        self.ts_id_name = config.ts_id_name
        self.ts_row_ranges = None
        self.time_period = None
        self.features_to_take = config.features_to_take
        self.indices_of_features_to_take_no_ids = config.indices_of_features_to_take_no_ids
        self.include_time = config.include_time
        self.include_ts_id = config.include_ts_id
        self.time_format = config.time_format
        self.preprocess_order: list[PreprocessNote] = None

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
