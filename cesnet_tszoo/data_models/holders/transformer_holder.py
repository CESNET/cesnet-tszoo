from dataclasses import dataclass, field
from copy import copy

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.transformer import Transformer


@dataclass
class TransformerHolder(Holder):
    transformers: list[Transformer] | Transformer = field(init=True)
    is_transformer_per_time_series: bool = field(init=True)
    should_partial_fit: bool = field(init=True)

    def get_instance(self, idx: int) -> list[Transformer] | Transformer:

        if self.transformers is None:
            raise ValueError()

        if self.is_transformer_per_time_series and idx > len(self.transformers):
            raise ValueError()

        return self.transformers[idx] if self.is_transformer_per_time_series else self.transformers

    def create_split_copy(self, split_range: slice) -> "TransformerHolder":

        split_copy = copy(self)

        split_copy.transformers = list(self.transformers[split_range]) if self.is_transformer_per_time_series else self.transformers

        return split_copy
