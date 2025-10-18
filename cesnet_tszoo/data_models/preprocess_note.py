from dataclasses import dataclass, field
from copy import copy

from cesnet_tszoo.utils.enums import PreprocessType
from cesnet_tszoo.data_models.holders.holder import Holder


@dataclass
class PreprocessNote:

    preprocess_type: PreprocessType = field(init=True)
    should_be_fitted: bool = field(init=True)
    is_inner_preprocess: bool = field(init=True)
    holder: Holder = field(init=True)

    def get_from_holder(self, idx: int) -> object:
        return self.holder.get_instance(idx)

    def create_split_copy(self, split_range: slice) -> "PreprocessNote":
        """Creates copy with splitted values. """

        split_copy = copy(self)

        split_copy.holder = self.holder.create_split_copy(split_range)

        return split_copy
