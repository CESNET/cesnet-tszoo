import numpy as np

from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.data_models.init_dataset_configs.init_config import DatasetInitConfig


class TimeDatasetInitConfig(DatasetInitConfig):
    """For time based init datasets. """

    def __init__(self, config: TimeBasedConfig):
        self.config = config

        self.ts_row_ranges = config.ts_row_ranges
        self.ts_ids = config.ts_ids

        self.train_time_period = None
        self.val_time_period = None
        self.test_time_period = None
        self.all_time_period = None

        self.train_preprocess_order_group = None
        self.val_preprocess_order_group = None
        self.test_preprocess_order_group = None

        super().__init__(config, None)

        if self.all_time_period is None:
            for temp_time_period in [self.train_time_period, self.val_time_period, self.test_time_period]:
                if temp_time_period is None:
                    continue
                elif self.time_period is None:
                    self.time_period = temp_time_period
                else:
                    self.time_period = np.concatenate((self.time_period, temp_time_period))

            self.time_period = np.unique(self.time_period)
        else:
            self.time_period = self.all_time_period

    def _init_train(self):
        """Initializes from train data of config """

        self.train_time_period = self.config.train_time_period
        self.train_preprocess_order_group = self.config.train_preprocess_order

    def _init_val(self):
        """Initializes from val data of config """

        self.val_time_period = self.config.val_time_period
        self.train_preprocess_order_group = self.config.val_preprocess_order

    def _init_test(self):
        """Initializes from test data of config """

        self.test_time_period = self.config.test_time_period
        self.train_preprocess_order_group = self.config.test_preprocess_order

    def _init_all(self):
        """Initializes from all data of config """

        self.all_time_period = self.config.all_time_period
        self.train_preprocess_order_group = self.config.all_preprocess_order
