from dataclasses import dataclass, field

from cesnet_tszoo.utils.enums import PreprocessType
from cesnet_tszoo.data_models.holders.holder import Holder


@dataclass
class PreprocessNote:

    preprocess_type: PreprocessType = field(init=True)
    should_be_fitted: bool = field(init=True)
    is_inner_preprocess: bool = field(init=True)
    args: Holder = field(init=True)
