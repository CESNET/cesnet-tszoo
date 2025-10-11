from cesnet_tszoo.configs import SeriesBasedConfig
from cesnet_tszoo.data_models.init_dataset_configs.init_config import DatasetInitConfig
from cesnet_tszoo.utils.enums import SplitType


class SeriesDatasetInitConfig(DatasetInitConfig):
    """For series based init datasets. """

    def __init__(self, config: SeriesBasedConfig, limit_init_to_set: SplitType):
        self.config = config

        self.ts_row_ranges = None
        self.ts_ids = None
        self.fillers = None
        self.preprocess_order_group = None

        super().__init__(config, limit_init_to_set)

        self.time_period = self.config.time_period

    def _init_train(self):
        """Initializes from train data of config """

        self.ts_row_ranges = self.config.train_ts_row_ranges
        self.ts_ids = self.config.train_ts
        self.preprocess_order_group = self.config.train_preprocess_order

    def _init_val(self):
        """Initializes from val data of config """

        self.ts_row_ranges = self.config.val_ts_row_ranges
        self.ts_ids = self.config.val_ts
        self.preprocess_order_group = self.config.val_preprocess_order

    def _init_test(self):
        """Initializes from test data of config """

        self.ts_row_ranges = self.config.test_ts_row_ranges
        self.ts_ids = self.config.test_ts
        self.preprocess_order_group = self.config.test_preprocess_order

    def _init_all(self):
        """Initializes from all data of config """

        self.ts_row_ranges = self.config.all_ts_row_ranges
        self.ts_ids = self.config.all_ts
        self.preprocess_order_group = self.config.all_preprocess_order
