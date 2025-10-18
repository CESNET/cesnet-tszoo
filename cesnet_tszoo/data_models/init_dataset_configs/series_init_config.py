from copy import deepcopy

from cesnet_tszoo.configs import SeriesBasedConfig
from cesnet_tszoo.data_models.init_dataset_configs.init_config import DatasetInitConfig
from cesnet_tszoo.data_models.preprocess_order_group import PreprocessOrderGroup
from cesnet_tszoo.utils.enums import SplitType


class SeriesDatasetInitConfig(DatasetInitConfig):
    """For series based init datasets. """

    def __init__(self, config: SeriesBasedConfig, limit_init_to_set: SplitType, group: PreprocessOrderGroup):
        self.ts_row_ranges = None
        self.ts_ids = None
        self.preprocess_order_group = deepcopy(group)

        super().__init__(config, limit_init_to_set)

        self.time_period = config.time_period

    def _init_train(self, config: SeriesBasedConfig):
        """Initializes from train data of config """

        self.ts_row_ranges = config.train_ts_row_ranges
        self.ts_ids = config.train_ts

    def _init_val(self, config: SeriesBasedConfig):
        """Initializes from val data of config """

        self.ts_row_ranges = config.val_ts_row_ranges
        self.ts_ids = config.val_ts

    def _init_test(self, config: SeriesBasedConfig):
        """Initializes from test data of config """

        self.ts_row_ranges = config.test_ts_row_ranges
        self.ts_ids = config.test_ts

    def _init_all(self, config: SeriesBasedConfig):
        """Initializes from all data of config """

        self.ts_row_ranges = config.all_ts_row_ranges
        self.ts_ids = config.all_ts
