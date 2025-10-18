from typing import Optional
from copy import copy, deepcopy

from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.data_models.load_dataset_configs.load_config import LoadConfig
from cesnet_tszoo.utils.enums import SplitType


class TimeLoadConfig(LoadConfig):
    """Base class for dataset configs that are used to pass values to load datasets. """

    def __init__(self, config: TimeBasedConfig, limit_init_to_set: Optional[SplitType]):
        self.config = config

        super().__init__(config, limit_init_to_set)

        self.ts_row_ranges = config.ts_row_ranges

    def _init_train(self, config: TimeBasedConfig):
        """Initializes from train data of config """

        self.time_period = config.train_time_period
        self.fillers = deepcopy(config.train_fillers)
        self.anomaly_handlers = config.anomaly_handlers

    def _init_val(self, config: TimeBasedConfig):
        """Initializes from val data of config """

        self.time_period = config.val_time_period
        self.fillers = deepcopy(config.val_fillers)

    def _init_test(self, config: TimeBasedConfig):
        """Initializes from test data of config """

        self.time_period = config.test_time_period
        self.fillers = deepcopy(config.test_fillers)

    def _init_all(self, config: TimeBasedConfig):
        """Initializes from all data of config """

        self.time_period = config.all_time_period
        self.fillers = deepcopy(config.all_fillers)

    def create_split_copy(self, split_range: slice):
        """Creates copy with splitted values. """
        split_copy = copy(self)

        split_copy.transformers = self.transformers[split_range] if self.is_transformer_per_time_series else self.transformers
        split_copy.fillers = deepcopy(self.fillers[split_range])
        split_copy.anomaly_handlers = None if self.anomaly_handlers is None else self.anomaly_handlers[split_range]
        split_copy.ts_row_ranges = self.ts_row_ranges[split_range]

        return split_copy
