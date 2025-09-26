from typing import Optional
from copy import deepcopy, copy

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
        self.fillers = deepcopy(self.config.train_fillers)
        self.anomaly_handlers = self.config.anomaly_handlers

    def _init_val(self):
        """Initializes from val data of config """

        self.ts_row_ranges = self.config.val_ts_row_ranges
        self.fillers = deepcopy(self.config.val_fillers)

    def _init_test(self):
        """Initializes from test data of config """

        self.ts_row_ranges = self.config.test_ts_row_ranges
        self.fillers = deepcopy(self.config.test_fillers)

    def _init_all(self):
        """Initializes from all data of config """

        self.ts_row_ranges = self.config.all_ts_row_ranges
        self.fillers = deepcopy(self.config.all_fillers)

    def create_split_copy(self, split_range: slice):
        """Creates copy with splitted values. """

        split_copy = copy(self)

        split_copy.transformers = self.transformers[split_range] if self.is_transformer_per_time_series else self.transformers
        split_copy.fillers = deepcopy(self.fillers[split_range])
        split_copy.anomaly_handlers = None if self.anomaly_handlers is None else self.anomaly_handlers[split_range]
        split_copy.ts_row_ranges = self.ts_row_ranges[split_range]

        return split_copy
