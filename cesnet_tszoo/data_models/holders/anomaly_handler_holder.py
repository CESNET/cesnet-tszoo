from dataclasses import dataclass, field

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.anomaly_handler import AnomalyHandler


@dataclass
class AnomalyHandlerHolder(Holder):
    anomaly_handlers: list[AnomalyHandler] = field(init=True)
