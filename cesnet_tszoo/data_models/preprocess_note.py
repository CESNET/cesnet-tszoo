from dataclasses import dataclass, field

from cesnet_tszoo.utils.enums import PreprocessType


@dataclass
class PreprocessNote:

    preprocess_type: PreprocessType = field(init=True)
    should_be_fitted: bool = field(init=True)
    is_inner_preprocess: bool = field(init=True)
