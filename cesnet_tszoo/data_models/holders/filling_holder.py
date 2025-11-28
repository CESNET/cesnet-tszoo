from dataclasses import dataclass, field
from copy import copy

import numpy as np

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.filler import Filler


@dataclass
class FillingHolder(Holder):
    fillers: np.ndarray[Filler] = field(init=True)
    default_values: list[float] = field(init=True)

    def get_instance(self, idx: int) -> Filler:

        if self.fillers is None:
            raise ValueError()

        if idx > len(self.fillers):
            raise ValueError()

        return self.fillers[idx]

    def is_empty(self):
        return self.fillers is None

    def create_split_copy(self, split_range: slice) -> "FillingHolder":

        split_copy = copy(self)

        split_copy.fillers = np.array(self.fillers[split_range])

        return split_copy

    def apply(self, data: np.ndarray, idx: int, **kwargs) -> np.ndarray:

        mask = np.isnan(data)
        data[mask] = np.take(self.default_values, np.nonzero(mask)[1])

        self.get_instance(idx).fill(data, mask, default_values=self.default_values)

        return data

    def update_instance(self, update_with: Filler, idx: int):
        self.fillers[idx] = update_with

    def fit(self, data: np.ndarray, idx: int):
        ...

    def supported_ts_updated(self, supported_ts_ids: np.ndarray):
        if self.is_empty():
            return

        self.fillers = self.fillers[supported_ts_ids]
