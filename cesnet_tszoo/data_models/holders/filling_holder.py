from dataclasses import dataclass, field
from copy import copy

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.filler import Filler


@dataclass
class FillingHolder(Holder):
    fillers: list[Filler] = field(init=True)
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

        split_copy.fillers = list(self.fillers[split_range])

        return split_copy
