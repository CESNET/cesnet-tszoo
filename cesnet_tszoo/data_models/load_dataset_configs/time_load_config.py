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

    def _init_train(self):
        """Initializes from train data of config """

        self.time_period = self.config.train_time_period
        self.fillers = deepcopy(self.config.train_fillers)
        self.anomaly_handlers = self.config.anomaly_handlers

    def _init_val(self):
        """Initializes from val data of config """

        self.time_period = self.config.val_time_period
        self.fillers = deepcopy(self.config.val_fillers)

    def _init_test(self):
        """Initializes from test data of config """

        self.time_period = self.config.test_time_period
        self.fillers = deepcopy(self.config.test_fillers)

    def _init_all(self):
        """Initializes from all data of config """

        self.time_period = self.config.all_time_period
        self.fillers = deepcopy(self.config.all_fillers)

    def create_split_copy(self, split_range: slice):
        """Creates copy with splitted values. """
        split_copy = copy(self)

        split_copy.transformers = self.transformers[split_range] if self.is_transformer_per_time_series else self.transformers
        split_copy.fillers = deepcopy(self.fillers[split_range])
        split_copy.anomaly_handlers = None if self.anomaly_handlers is None else self.anomaly_handlers[split_range]
        split_copy.ts_row_ranges = self.ts_row_ranges[split_range]

        return split_copy
