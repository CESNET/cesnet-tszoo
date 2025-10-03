from abc import ABC, abstractmethod
from typing import Optional

from cesnet_tszoo.configs.base_config import DatasetConfig
from cesnet_tszoo.utils.enums import SplitType


class DatasetInitConfig(ABC):
    """Base class for dataset init configs that are used to pass values to init datasets. """

    def __init__(self, config: DatasetConfig, limit_init_to_set: Optional[SplitType]):
        self.config = config

        self.ts_id_name = config.ts_id_name
        self.features_to_take = config.features_to_take
        self.indices_of_features_to_take_no_ids = config.indices_of_features_to_take_no_ids
        self.default_values = config.default_values
        self.time_period = None
        self.anomaly_handlers = None

        if limit_init_to_set == SplitType.TRAIN:
            self._init_train()
        elif limit_init_to_set == SplitType.VAL:
            self._init_val()
        elif limit_init_to_set == SplitType.TEST:
            self._init_test()
        elif limit_init_to_set == SplitType.ALL:
            self._init_all()
        else:
            self._init_train()
            self._init_val()
            self._init_test()
            self._init_all()

    @abstractmethod
    def _init_train(self):
        """Initializes from train data of config """
        ...

    @abstractmethod
    def _init_val(self):
        """Initializes from val data of config """
        ...

    @abstractmethod
    def _init_test(self):
        """Initializes from test data of config """
        ...

    @abstractmethod
    def _init_all(self):
        """Initializes from all data of config """
        ...
