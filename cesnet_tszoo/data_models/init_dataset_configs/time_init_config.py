from copy import deepcopy

import numpy as np

from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.data_models.init_dataset_configs.init_config import DatasetInitConfig
from cesnet_tszoo.data_models.preprocess_order_group import PreprocessOrderGroup


class TimeDatasetInitConfig(DatasetInitConfig):
    """For time based init datasets. """

    def __init__(self, config: TimeBasedConfig, ts_ids_ignore: np.ndarray, train_group: PreprocessOrderGroup, val_group: PreprocessOrderGroup, test_group: PreprocessOrderGroup):
        self.ts_row_ranges = config.ts_row_ranges
        self.ts_ids = config.ts_ids

        self.train_time_period = None
        self.val_time_period = None
        self.test_time_period = None
        self.all_time_period = None
        self.ts_ids_ignore = ts_ids_ignore

        self.train_preprocess_order_group = deepcopy(train_group)
        self.val_preprocess_order_group = deepcopy(val_group)
        self.test_preprocess_order_group = deepcopy(test_group)

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

    def _init_train(self, config: TimeBasedConfig):
        """Initializes from train data of config """

        self.train_time_period = config.train_time_period

    def _init_val(self, config: TimeBasedConfig):
        """Initializes from val data of config """

        self.val_time_period = config.val_time_period

    def _init_test(self, config: TimeBasedConfig):
        """Initializes from test data of config """

        self.test_time_period = config.test_time_period

    def _init_all(self, config: TimeBasedConfig):
        """Initializes from all data of config """

        self.all_time_period = config.all_time_period
