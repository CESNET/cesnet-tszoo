import numpy as np

from cesnet_tszoo.pytables_data.base_datasets.initializer_dataset import InitializerDataset
from cesnet_tszoo.data_models.fitted_preprocess_instance import FittedPreprocessInstance
from cesnet_tszoo.data_models.init_dataset_configs.series_init_config import SeriesDatasetInitConfig
from cesnet_tszoo.data_models.holders import FillingHolder, TransformerHolder, AnomalyHandlerHolder
from cesnet_tszoo.data_models.init_dataset_return import InitDatasetReturn


class SeriesBasedInitializerDataset(InitializerDataset):
    """Used for series based datasets. Used for going through data to fit transformers, prepare fillers and validate thresholds."""

    def __init__(self, database_path: str, table_data_path: str, init_config: SeriesDatasetInitConfig):
        self.init_config = init_config

        super(SeriesBasedInitializerDataset, self).__init__(database_path, table_data_path, init_config)

    def __getitem__(self, idx) -> InitDatasetReturn:

        data, existing_indices = self.load_data_from_table(self.init_config.ts_row_ranges[idx])
        is_under_nan_threshold = len(existing_indices) / len(self.init_config.time_period) <= self.init_config.nan_threshold

        preprocess_fitted_instances = []
        train_data = np.array([])

        if is_under_nan_threshold:
            data = self._handle_data_preprocess(data, idx)

            for preprocess_order in self.init_config.preprocess_order_group.preprocess_inner_orders:
                if preprocess_order.should_be_fitted:
                    preprocess_fitted_instances.append(FittedPreprocessInstance(preprocess_order.preprocess_type, preprocess_order.get_from_holder(idx)))

            # Prepare data from current time series for training transformer if needed
            if len(self.init_config.indices_of_features_to_take_no_ids) == 1:
                train_data = data[: len(self.init_config.time_period), self.offset_exclude_feature_ids:].reshape(-1, 1)
            elif len(self.init_config.time_period) == 1:
                train_data = data[: len(self.init_config.time_period), self.offset_exclude_feature_ids:].reshape(1, -1)
            else:
                train_data = data[: len(self.init_config.time_period), self.offset_exclude_feature_ids:]

        return InitDatasetReturn(train_data, is_under_nan_threshold, preprocess_fitted_instances)

    def __len__(self) -> int:
        return len(self.init_config.ts_row_ranges)

    def _handle_data_preprocess(self, data: np.ndarray, idx: int) -> np.ndarray:
        return self._handle_data_preprocess_order_group(self.init_config.preprocess_order_group, data, idx)

    def _handle_filling(self, filling_holder: FillingHolder, data: np.ndarray, idx: int) -> None:
        """Just fills data. """

        mask = np.isnan(data[:, self.offset_exclude_feature_ids:])
        data[:, self.offset_exclude_feature_ids:][mask] = np.take(filling_holder.default_values, np.nonzero(mask)[1])

        filling_holder.fillers[idx].fill(data[:, self.offset_exclude_feature_ids:].view(), mask, default_values=filling_holder.default_values)

    def _handle_anomalies(self, anomaly_handler_holder: AnomalyHandlerHolder, should_fit: bool, data: np.ndarray, idx: int):
        """Fits and uses anomaly handlers. """

        if should_fit:
            anomaly_handler_holder.anomaly_handlers[idx].fit(data[:, self.offset_exclude_feature_ids:])

        anomaly_handler_holder.anomaly_handlers[idx].transform_anomalies(data[:, self.offset_exclude_feature_ids:])

    def _handle_transforming(self, transfomer_holder: TransformerHolder, should_fit: bool, data: np.ndarray, idx: int) -> np.ndarray:
        """Fits and uses transformers. """

        if should_fit and transfomer_holder.should_partial_fit:
            transfomer_holder.transformers.partial_fit(data[:, self.offset_exclude_feature_ids:])
        elif should_fit:
            transfomer_holder.transformers.fit(data[:, self.offset_exclude_feature_ids:])

        return transfomer_holder.transformers.transform(data[:, self.offset_exclude_feature_ids:])
