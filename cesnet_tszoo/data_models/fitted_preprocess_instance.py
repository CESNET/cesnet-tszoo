from dataclasses import dataclass, field

from cesnet_tszoo.utils.enums import PreprocessType


@dataclass
class FittedPreprocessInstance:
    preprocess_type: PreprocessType = field(init=True)
    instance: object = field(init=True)
