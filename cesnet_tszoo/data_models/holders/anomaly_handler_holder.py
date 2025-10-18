from dataclasses import dataclass, field
from copy import copy

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.anomaly_handler import AnomalyHandler


@dataclass
class AnomalyHandlerHolder(Holder):
    anomaly_handlers: list[AnomalyHandler] = field(init=True)

    def get_instance(self, idx: int) -> AnomalyHandler:

        if self.anomaly_handlers is None:
            raise ValueError()

        if idx > len(self.anomaly_handlers):
            raise ValueError()

        return self.anomaly_handlers[idx]

    def create_split_copy(self, split_range: slice) -> "AnomalyHandlerHolder":

        split_copy = copy(self)

        if self.anomaly_handlers is not None:
            split_copy.anomaly_handlers = list(self.anomaly_handlers[split_range])

        return split_copy
