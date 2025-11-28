from dataclasses import dataclass, field
from copy import copy

import numpy as np

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.transformer import Transformer


@dataclass
class TransformerHolder(Holder):
    transformers: np.ndarray[Transformer] | Transformer = field(init=True)
    is_transformer_per_time_series: bool = field(init=True)
    should_partial_fit: bool = field(init=True)

    def get_instance(self, idx: int) -> np.ndarray[Transformer] | Transformer:

        if self.transformers is None:
            raise ValueError()

        if self.is_transformer_per_time_series and idx > len(self.transformers):
            raise ValueError()

        return self.transformers[idx] if self.is_transformer_per_time_series else self.transformers

    def is_empty(self):
        return self.transformers is None

    def create_split_copy(self, split_range: slice) -> "TransformerHolder":

        split_copy = copy(self)

        split_copy.transformers = np.array(self.transformers[split_range]) if self.is_transformer_per_time_series else self.transformers

        return split_copy

    def apply(self, data: np.ndarray, idx: int, **kwargs) -> np.ndarray:
        return self.get_instance(idx).transform(data)

    def update_instance(self, update_with: Transformer, idx: int):
        if self.is_transformer_per_time_series:
            self.transformers[idx] = update_with
        else:
            self.transformers = update_with

    def fit(self, data: np.ndarray, idx: int):
        if self.should_partial_fit:
            self.get_instance(idx).partial_fit(data)
        else:
            self.get_instance(idx).fit(data)

    def supported_ts_updated(self, supported_ts_ids: np.ndarray):
        if self.is_empty():
            return

        if self.is_transformer_per_time_series:
            self.transformers = self.transformers[supported_ts_ids]
