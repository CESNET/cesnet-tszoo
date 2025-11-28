from typing import Optional
from copy import copy, deepcopy

from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.data_models.load_dataset_configs.load_config import LoadConfig
from cesnet_tszoo.utils.enums import SplitType


class TimeLoadConfig(LoadConfig):
    """Base class for dataset configs that are used to pass values to load datasets. """

    def __init__(self, config: TimeBasedConfig, limit_init_to_set: Optional[SplitType]):
        super().__init__(config, limit_init_to_set)

        self.ts_row_ranges = config.ts_row_ranges

    def _init_train(self, config: TimeBasedConfig):
        """Initializes from train data of config """

        self.time_period = config.train_time_period

        self.preprocess_order = deepcopy(config.train_preprocess_order)

    def _init_val(self, config: TimeBasedConfig):
        """Initializes from val data of config """

        self.time_period = config.val_time_period

        self.preprocess_order = deepcopy(config.val_preprocess_order)

    def _init_test(self, config: TimeBasedConfig):
        """Initializes from test data of config """

        self.time_period = config.test_time_period

        self.preprocess_order = deepcopy(config.test_preprocess_order)

    def _init_all(self, config: TimeBasedConfig):
        """Initializes from all data of config """

        self.time_period = config.all_time_period

        self.preprocess_order = deepcopy(config.all_preprocess_order)

    def create_split_copy(self, split_range: slice):
        """Creates copy with splitted values. """
        split_copy = copy(self)

        split_copy.preprocess_order = [deepcopy(preprocess_note.create_split_copy(split_range)) for preprocess_note in self.preprocess_order]

        split_copy.ts_row_ranges = self.ts_row_ranges[split_range]

        return split_copy
