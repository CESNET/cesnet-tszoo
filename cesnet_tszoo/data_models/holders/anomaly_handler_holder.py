from dataclasses import dataclass, field

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
