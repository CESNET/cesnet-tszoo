from dataclasses import dataclass, field
from copy import copy

import numpy as np

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.anomaly_handler import AnomalyHandler


@dataclass
class AnomalyHandlerHolder(Holder):
    anomaly_handlers: np.ndarray[AnomalyHandler] = field(init=True)

    def get_instance(self, idx: int) -> AnomalyHandler:

        if self.anomaly_handlers is None:
            raise ValueError()

        if idx > len(self.anomaly_handlers):
            raise ValueError()

        return self.anomaly_handlers[idx]

    def is_empty(self):
        return self.anomaly_handlers is None

    def create_split_copy(self, split_range: slice) -> "AnomalyHandlerHolder":

        split_copy = copy(self)

        if self.anomaly_handlers is not None:
            split_copy.anomaly_handlers = np.array(self.anomaly_handlers[split_range])

        return split_copy

    def update_instance(self, update_with: AnomalyHandler, idx: int):
        self.anomaly_handlers[idx] = update_with

    def fit(self, data: np.ndarray, idx: int):
        self.get_instance(idx).fit(data)

    def supported_ts_updated(self, supported_ts_ids: np.ndarray):
        if self.is_empty():
            return

        self.anomaly_handlers = self.anomaly_handlers[supported_ts_ids]

    def apply(self, data: np.ndarray, idx: int, **kwargs) -> np.ndarray:
        self.get_instance(idx).transform_anomalies(data)

        return data
