from cesnet_tszoo.configs import DisjointTimeBasedConfig
from cesnet_tszoo.data_models.init_dataset_configs.init_config import DatasetInitConfig
from cesnet_tszoo.utils.enums import SplitType


class DisjointTimeDatasetInitConfig(DatasetInitConfig):
    """For disjoint time based init datasets. """

    def __init__(self, config: DisjointTimeBasedConfig, limit_init_to_set: SplitType):
        self.config = config

        self.ts_row_ranges = None
        self.ts_ids = None

        self.preprocess_order_group = None

        super().__init__(config, limit_init_to_set)

    def _init_train(self):
        """Initializes from train data of config """

        self.time_period = self.config.train_time_period
        self.ts_row_ranges = self.config.train_ts_row_ranges
        self.ts_ids = self.config.train_ts
        self.preprocess_order_group = self.config.train_preprocess_order

    def _init_val(self):
        """Initializes from val data of config """

        self.time_period = self.config.val_time_period
        self.ts_row_ranges = self.config.val_ts_row_ranges
        self.ts_ids = self.config.val_ts
        self.preprocess_order_group = self.config.val_preprocess_order

    def _init_test(self):
        """Initializes from test data of config """

        self.time_period = self.config.test_time_period
        self.ts_row_ranges = self.config.test_ts_row_ranges
        self.ts_ids = self.config.test_ts
        self.preprocess_order_group = self.config.test_preprocess_order

    def _init_all(self):
        """Initializes from all data of config """
        raise NotImplementedError()
