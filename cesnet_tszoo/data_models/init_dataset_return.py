from dataclasses import dataclass, field

from cesnet_tszoo.data_models.fitted_preprocess_instance import FittedPreprocessInstance


@dataclass
class InitDatasetReturn:
    train_data: bool = field(init=True)
    is_under_nan_threshold: bool = field(init=True)
    preprocess_fitted_instances: list[FittedPreprocessInstance] = field(init=True)
