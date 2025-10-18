from copy import deepcopy

from cesnet_tszoo.configs import DisjointTimeBasedConfig
from cesnet_tszoo.data_models.init_dataset_configs.init_config import DatasetInitConfig
from cesnet_tszoo.utils.enums import SplitType
from cesnet_tszoo.data_models.preprocess_order_group import PreprocessOrderGroup


class DisjointTimeDatasetInitConfig(DatasetInitConfig):
    """For disjoint time based init datasets. """

    def __init__(self, config: DisjointTimeBasedConfig, limit_init_to_set: SplitType, group: PreprocessOrderGroup):
        self.ts_row_ranges = None
        self.ts_ids = None
        self.preprocess_order_group = deepcopy(group)

        super().__init__(config, limit_init_to_set)

    def _init_train(self, config: DisjointTimeBasedConfig):
        """Initializes from train data of config """

        self.time_period = config.train_time_period
        self.ts_row_ranges = config.train_ts_row_ranges
        self.ts_ids = config.train_ts

    def _init_val(self, config: DisjointTimeBasedConfig):
        """Initializes from val data of config """

        self.time_period = config.val_time_period
        self.ts_row_ranges = config.val_ts_row_ranges
        self.ts_ids = config.val_ts

    def _init_test(self, config: DisjointTimeBasedConfig):
        """Initializes from test data of config """

        self.time_period = config.test_time_period
        self.ts_row_ranges = config.test_ts_row_ranges
        self.ts_ids = config.test_ts

    def _init_all(self, config: DisjointTimeBasedConfig):
        """Initializes from all data of config """
        raise NotImplementedError()
