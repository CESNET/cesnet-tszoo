from dataclasses import dataclass, field

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.transformer import Transformer


@dataclass
class TransformerHolder(Holder):
    transformers: list[Transformer] | Transformer = field(init=True)
    is_transformer_per_time_series: bool = field(init=True)
    should_partial_fit: bool = field(init=True)
