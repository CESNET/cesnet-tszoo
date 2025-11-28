from abc import ABC, abstractmethod
import inspect

from cesnet_tszoo.utils.enums import PreprocessType, SplitType
from cesnet_tszoo.utils.custom_handler.custom_handler import CustomHandler, PerSeriesCustomHandler, AllSeriesCustomHandler, NoFitCustomHandler


class CustomHandlerFactory(ABC):
    def __init__(self, base_handler_type: type[CustomHandler], can_fit: bool, can_partial_fit: bool, preprocess_enum_type: PreprocessType, is_per_series: bool):
        self.name = None
        self.handler_type = None
        self.can_fit = can_fit
        self.can_partial_fit = can_partial_fit
        self.base_handler_type = base_handler_type
        self.is_per_series = is_per_series
        self.target_sets = []
        self.preprocess_enum_type = preprocess_enum_type
        self.can_apply_to_train = False
        self.can_apply_to_val = False
        self.can_apply_to_test = False
        self.can_apply_to_all = False

    @abstractmethod
    def create_handler(self) -> CustomHandler:
        ...

    def post_init(self, handler_type: type[CustomHandler]):
        self.handler_type = handler_type
        self.target_sets = handler_type.get_target_sets()

        if len(self.target_sets) == 0:
            raise ValueError(f"At least one set must be targeted ({[SplitType.TRAIN, SplitType.VAL, SplitType.TEST, SplitType.ALL]})")

        for target_set in self.target_sets:
            target_set = SplitType(target_set)

            if target_set == SplitType.TRAIN:
                self.can_apply_to_train = True
            elif target_set == SplitType.VAL:
                self.can_apply_to_val = True
            elif target_set == SplitType.TEST:
                self.can_apply_to_test = True
            elif target_set == SplitType.ALL:
                self.can_apply_to_all = True

        if self.can_be_used(handler_type):
            self.name = self.handler_type.__name__

    def can_be_used(self, handler_type: type) -> bool:
        return isinstance(handler_type, type) and inspect.isclass(handler_type) and issubclass(handler_type, self.base_handler_type)


# Implemented factories

class PerSeriesCustomHandlerFactory(CustomHandlerFactory):
    def __init__(self):
        super().__init__(PerSeriesCustomHandler, True, False, PreprocessType.PER_SERIES_CUSTOM, True)

    def create_handler(self) -> PerSeriesCustomHandler:
        return self.handler_type()


class AllSeriesCustomHandlerFactory(CustomHandlerFactory):
    def __init__(self):
        super().__init__(AllSeriesCustomHandler, True, False, PreprocessType.ALL_SERIES_CUSTOM, False)

    def create_handler(self) -> AllSeriesCustomHandler:
        return self.handler_type()


class NoFitCustomHandlerFactory(CustomHandlerFactory):
    def __init__(self):
        super().__init__(NoFitCustomHandler, False, False, PreprocessType.NO_FIT_CUSTOM, False)

    def create_handler(self) -> NoFitCustomHandler:
        return self.handler_type()

# Implemented factories


def get_custom_handler_factory(handler_type: type) -> CustomHandlerFactory:

    for factory in CustomHandlerFactory.__subclasses__():
        factory_instance = factory()
        factory_instance.post_init(handler_type)

        if factory_instance.can_be_used(handler_type):
            return factory_instance

    raise TypeError(f"Passed custom handler {handler_type} cannot be used. It must be passed as type and derive from one of the classes: {[PerSeriesCustomHandler, AllSeriesCustomHandler, NoFitCustomHandler]}")
