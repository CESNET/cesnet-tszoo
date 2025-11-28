from dataclasses import dataclass, field
from copy import copy

from cesnet_tszoo.data_models.preprocess_note import PreprocessNote


@dataclass
class PreprocessOrderGroup:
    """Represent preprocess groups. Every preprocess in this group will be done in one init loop. """

    preprocess_orders: list[PreprocessNote] = field(init=True)
    preprocess_inner_orders: list[PreprocessNote] = field(init=False, default_factory=lambda: [])
    preprocess_outer_orders: list[PreprocessNote] = field(init=False, default_factory=lambda: [])
    any_preprocess_needs_fitting: bool = field(init=False, default=False)
    any_preprocess_is_dummy_fitting: bool = field(init=False, default=False)

    def __post_init__(self):

        for preprocess in self.preprocess_orders:
            if preprocess.is_inner_preprocess:
                self.preprocess_inner_orders.append(preprocess)
            else:
                self.preprocess_outer_orders.append(preprocess)

            self.any_preprocess_needs_fitting = True if preprocess.should_be_fitted else self.any_preprocess_needs_fitting
            self.any_preprocess_is_dummy_fitting = True if preprocess.is_dummy_should_be_fitted else self.any_preprocess_is_dummy_fitting

    def get_preprocess_orders_for_inner_transform(self) -> list[PreprocessNote]:
        preprocess_orders_copy = [copy(preprocess_order) for preprocess_order in self.preprocess_orders]

        for preprocess_order in preprocess_orders_copy:
            preprocess_order.is_inner_preprocess = True
            preprocess_order.should_be_fitted = False
            preprocess_order.is_dummy_should_be_fitted = False

        return preprocess_orders_copy
