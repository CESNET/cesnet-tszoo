from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.utils.enums import SplitType


class DatasetInitConfig(ABC):
    """Base class for dataset init configs that are used to pass values to init datasets. """

    def __init__(self, config: DatasetConfig, ts_ids_ignore: np.ndarray, limit_init_to_set: Optional[SplitType]):
        self.ts_id_name = config.ts_id_name
        self.features_to_take = config.features_to_take
        self.indices_of_features_to_take_no_ids = config.indices_of_features_to_take_no_ids
        self.time_period = None
        self.nan_threshold = config.nan_threshold
        self.ts_ids_ignore = ts_ids_ignore

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
