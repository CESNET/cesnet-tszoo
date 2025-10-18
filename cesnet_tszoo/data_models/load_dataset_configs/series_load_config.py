from typing import Optional
from copy import copy, deepcopy

from cesnet_tszoo.configs import SeriesBasedConfig
from cesnet_tszoo.data_models.load_dataset_configs.load_config import LoadConfig
from cesnet_tszoo.utils.enums import SplitType


class SeriesLoadConfig(LoadConfig):
    """Base class for dataset configs that are used to pass values to load datasets. """

    def __init__(self, config: SeriesBasedConfig, limit_init_to_set: Optional[SplitType]):
        self.config = config

        super().__init__(config, limit_init_to_set)

        self.time_period = self.config.time_period

    def _init_train(self):
        """Initializes from train data of config """

        self.ts_row_ranges = self.config.train_ts_row_ranges
        self.preprocess_order = deepcopy(self.config.train_preprocess_order)

    def _init_val(self):
        """Initializes from val data of config """

        self.ts_row_ranges = self.config.val_ts_row_ranges
        self.preprocess_order = deepcopy(self.config.val_preprocess_order)

    def _init_test(self):
        """Initializes from test data of config """

        self.ts_row_ranges = self.config.test_ts_row_ranges
        self.preprocess_order = deepcopy(self.config.test_preprocess_order)

    def _init_all(self):
        """Initializes from all data of config """

        self.ts_row_ranges = self.config.all_ts_row_ranges
        self.preprocess_order = deepcopy(self.config.all_preprocess_order)

    def create_split_copy(self, split_range: slice):
        """Creates copy with splitted values. """

        split_copy = copy(self)

        split_copy.preprocess_order = [deepcopy(preprocess_note.create_split_copy(split_range)) for preprocess_note in self.preprocess_order]

        split_copy.ts_row_ranges = self.ts_row_ranges[split_range]

        return split_copy
