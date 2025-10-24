from copy import deepcopy

import numpy as np

from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME
from cesnet_tszoo.pytables_data.base_datasets.initializer_dataset import InitializerDataset
from cesnet_tszoo.data_models.init_dataset_configs.time_init_config import TimeDatasetInitConfig
from cesnet_tszoo.data_models.fitted_preprocess_instance import FittedPreprocessInstance
from cesnet_tszoo.data_models.holders import FillingHolder, TransformerHolder, AnomalyHandlerHolder
from cesnet_tszoo.data_models.init_dataset_return import InitDatasetReturn
from cesnet_tszoo.utils.enums import PreprocessType
from cesnet_tszoo.data_models.preprocess_note import PreprocessNote


class TimeBasedInitializerDataset(InitializerDataset):
    """Used for time based datasets. Used for going through data to fit transformers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, init_config: TimeDatasetInitConfig):
        self.init_config = init_config

        super(TimeBasedInitializerDataset, self).__init__(database_path, table_data_path, init_config)

    def __getitem__(self, idx):

        data, existing_indices = self.load_data_from_table(self.init_config.ts_row_ranges[idx])

        offset = 0

        can_preprocess = True

        is_train_under_nan_threshold = True
        if self.init_config.train_time_period is not None and can_preprocess:
            in_train = (existing_indices < len(self.init_config.train_time_period)).sum()
            is_train_under_nan_threshold = (in_train - offset) / len(self.init_config.train_time_period) <= self.init_config.nan_threshold
            offset += in_train
        can_preprocess = can_preprocess and is_train_under_nan_threshold

        is_val_under_nan_threshold = True
        if self.init_config.val_time_period is not None and can_preprocess:
            in_val = (existing_indices < len(self.init_config.val_time_period)).sum()
            is_val_under_nan_threshold = (in_val - offset) / len(self.init_config.val_time_period) <= self.init_config.nan_threshold
            offset += in_val
        can_preprocess = can_preprocess and is_val_under_nan_threshold

        is_test_under_nan_threshold = True
        if self.init_config.test_time_period is not None and can_preprocess:
            in_test = (existing_indices < len(self.init_config.test_time_period)).sum()
            is_test_under_nan_threshold = (in_test - offset) / len(self.init_config.test_time_period) <= self.init_config.nan_threshold
            offset += in_test
        can_preprocess = can_preprocess and is_test_under_nan_threshold

        is_all_under_nan_threshold = True
        if self.init_config.all_time_period is not None and can_preprocess:
            in_all = (existing_indices < len(self.init_config.all_time_period)).sum()
            is_all_under_nan_threshold = (in_all - offset) / len(self.init_config.all_time_period) <= self.init_config.nan_threshold
        can_preprocess = can_preprocess and is_all_under_nan_threshold

        train_preprocess_fitted_instances = []
        val_preprocess_fitted_instances = []
        test_preprocess_fitted_instances = []

        train_data = np.array([])
        if can_preprocess:

            # Prepare data from current time series for training
            if len(self.init_config.indices_of_features_to_take_no_ids) == 1:
                train_data = data[:, self.offset_exclude_feature_ids:].reshape(-1, 1)
            elif len(self.init_config.time_period) == 1:
                train_data = data[:, self.offset_exclude_feature_ids:].reshape(1, -1)
            else:
                train_data = data[:, self.offset_exclude_feature_ids:]

            train_data = self._handle_data_preprocess(train_data, idx)

            if self.init_config.train_time_period is not None:
                train_data = train_data[: len(self.init_config.train_time_period)]
            else:
                train_data = np.array([])

            for preprocess_order in self.init_config.train_preprocess_order_group.preprocess_inner_orders:
                if preprocess_order.should_be_fitted and not preprocess_order.holder.is_empty():
                    train_preprocess_fitted_instances.append(FittedPreprocessInstance(preprocess_order.preprocess_type, preprocess_order.get_from_holder(idx)))

            for preprocess_order in self.init_config.val_preprocess_order_group.preprocess_inner_orders:
                if preprocess_order.should_be_fitted and not preprocess_order.holder.is_empty():
                    val_preprocess_fitted_instances.append(FittedPreprocessInstance(preprocess_order.preprocess_type, preprocess_order.get_from_holder(idx)))

            for preprocess_order in self.init_config.test_preprocess_order_group.preprocess_inner_orders:
                if preprocess_order.should_be_fitted and not preprocess_order.holder.is_empty():
                    test_preprocess_fitted_instances.append(FittedPreprocessInstance(preprocess_order.preprocess_type, preprocess_order.get_from_holder(idx)))

        return InitDatasetReturn(train_data, is_train_under_nan_threshold, train_preprocess_fitted_instances), InitDatasetReturn(None, is_val_under_nan_threshold, val_preprocess_fitted_instances), InitDatasetReturn(None, is_test_under_nan_threshold, test_preprocess_fitted_instances), InitDatasetReturn(None, is_all_under_nan_threshold, None)

    def __len__(self) -> int:
        return len(self.init_config.ts_row_ranges)

    def _handle_data_preprocess(self, data: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:

        train_preprocess_inner_orders = self.init_config.train_preprocess_order_group.preprocess_inner_orders
        val_preprocess_inner_orders = self.init_config.val_preprocess_order_group.preprocess_inner_orders
        test_preprocess_inner_orders = self.init_config.test_preprocess_order_group.preprocess_inner_orders

        for train_preprocess, val_preprocess, test_preprocess in list(zip(train_preprocess_inner_orders, val_preprocess_inner_orders, test_preprocess_inner_orders)):

            if train_preprocess.preprocess_type == PreprocessType.FILLING_GAPS:
                train_filling_holder = train_preprocess.holder
                val_filling_holder = val_preprocess.holder
                test_filling_holder = test_preprocess.holder

                need_val_fit = val_preprocess.should_be_fitted
                need_test_fit = test_preprocess.should_be_fitted

                self._handle_filling(train_filling_holder, val_filling_holder, test_filling_holder, need_val_fit, need_test_fit, data, idx)
                continue

            offset = 0
            if self.init_config.train_time_period is not None:
                data = self._handle_data_preprocess_train_order(train_preprocess, data, idx)
                offset += len(self.init_config.train_time_period)

            if self.init_config.val_time_period is not None:
                data[offset:len(self.init_config.val_time_period) + offset] = self._handle_data_preprocess_other_order(val_preprocess, data[offset:len(self.init_config.val_time_period) + offset], idx)
                offset += len(self.init_config.val_time_period)

            if self.init_config.test_time_period is not None:
                data[offset:len(self.init_config.test_time_period) + offset] = self._handle_data_preprocess_other_order(test_preprocess, data[offset:len(self.init_config.test_time_period) + offset], idx)
                offset += len(self.init_config.test_time_period)

        return data

    def _handle_data_preprocess_other_order(self, preprocess_order: PreprocessNote, data: np.ndarray, idx: int) -> np.ndarray:
        if preprocess_order.preprocess_type == PreprocessType.HANDLING_ANOMALIES:
            return data
        elif preprocess_order.preprocess_type == PreprocessType.TRANSFORMING:
            return data
        else:
            raise NotImplementedError()

    def _handle_data_preprocess_train_order(self, preprocess_order: PreprocessNote, data: np.ndarray, idx: int) -> np.ndarray:
        if preprocess_order.preprocess_type == PreprocessType.HANDLING_ANOMALIES:
            self._handle_anomalies(preprocess_order.holder, preprocess_order.should_be_fitted, data[:len(self.init_config.train_time_period)].view(), idx)
        elif preprocess_order.preprocess_type == PreprocessType.TRANSFORMING:
            data = self._handle_transforming(preprocess_order.holder, preprocess_order.should_be_fitted, data, idx)
        else:
            raise NotImplementedError()

        return data

    def _handle_filling(self, train_filling_holder: FillingHolder, val_filling_holder: FillingHolder, test_filling_holder: FillingHolder, need_val_fit: bool, need_test_fit: bool, data: np.ndarray, idx: int):
        """Fills data and prepares fillers based on previous times. Order is train (does not need to update train fillers) > val > test."""

        offset_start = np.inf
        first_start_index = None
        train_should_fill = True
        previous_offset = 0

        # Missing/existing indices for train set
        if self.init_config.train_time_period is not None:
            first_start_index = offset_start = self.init_config.train_time_period[ID_TIME_COLUMN_NAME][0]

        # Missing/existing indices for validation set; Additionally prepares validation fillers based on previous values if needed and fills data
        if self.init_config.val_time_period is not None and need_val_fit:

            current_start_index = self.init_config.val_time_period[ID_TIME_COLUMN_NAME][0]
            if first_start_index is not None and current_start_index > first_start_index:
                previous_offset = current_start_index - first_start_index

                val_mask = np.isnan(data[:current_start_index - first_start_index])
                data[:current_start_index - first_start_index][val_mask] = np.take(val_filling_holder.default_values, np.nonzero(val_mask)[1])

                train_should_fill = False
                val_filling_holder.get_instance(idx).fill(data[:current_start_index - first_start_index].view(), val_mask, default_values=val_filling_holder.default_values)

            offset_start = min(offset_start, self.init_config.val_time_period[ID_TIME_COLUMN_NAME][0])

            if first_start_index is None:
                first_start_index = current_start_index

        # Missing/existing indices for test set; Additionally prepares test fillers based on previous values if needed and fills data
        if self.init_config.test_time_period is not None and need_test_fit:

            current_start_index = self.init_config.test_time_period[ID_TIME_COLUMN_NAME][0]

            if first_start_index is not None and current_start_index > first_start_index:

                if self.init_config.val_time_period is not None:
                    test_filling_holder.fillers[idx] = deepcopy(val_filling_holder.get_instance(idx))

                test_mask = np.isnan(data[previous_offset:current_start_index - first_start_index])
                data[previous_offset:current_start_index - first_start_index][test_mask] = np.take(test_filling_holder.default_values, np.nonzero(test_mask)[1])

                train_should_fill = False
                test_filling_holder.get_instance(idx).fill(data[previous_offset:current_start_index - first_start_index].view(), test_mask, default_values=test_filling_holder.default_values)

            offset_start = min(offset_start, self.init_config.test_time_period[ID_TIME_COLUMN_NAME][0])

        if self.init_config.train_time_period is not None and train_should_fill:  # for transformer...
            train_mask = np.isnan(data)
            data[train_mask] = np.take(train_filling_holder.default_values, np.nonzero(train_mask)[1])

            train_filling_holder.get_instance(idx).fill(data.view(), train_mask, default_values=train_filling_holder.default_values)

    def _handle_transforming(self, transfomer_holder: TransformerHolder, should_fit: bool, data: np.ndarray, idx: int) -> np.ndarray:
        """Fits and uses transformers. """

        if len(self.init_config.indices_of_features_to_take_no_ids) == 1:
            train_data = data[: len(self.init_config.train_time_period), :].reshape(-1, 1)
        elif len(self.init_config.train_time_period) == 1:
            train_data = data[: len(self.init_config.train_time_period), :].reshape(1, -1)
        else:
            train_data = data[: len(self.init_config.train_time_period), :]

        if should_fit and transfomer_holder.should_partial_fit:
            transfomer_holder.get_instance(idx).partial_fit(train_data)
        elif should_fit:
            transfomer_holder.get_instance(idx).fit(train_data)

        return transfomer_holder.get_instance(idx).transform(data)

    def _handle_anomalies(self, anomaly_handler_holder: AnomalyHandlerHolder, should_fit: bool, data: np.ndarray, idx: int):
        """Fits and uses anomaly handlers. """

        if should_fit:
            anomaly_handler_holder.anomaly_handlers[idx].fit(data)

        anomaly_handler_holder.anomaly_handlers[idx].transform_anomalies(data.view())
