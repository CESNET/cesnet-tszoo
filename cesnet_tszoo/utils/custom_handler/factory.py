from abc import ABC, abstractmethod
import inspect

from cesnet_tszoo.utils.custom_handler.custom_handler import CustomHandler, PerSeriesCustomHandler, AllSeriesCustomHandler, NoFitCustomHandler


class CustomHandlerFactory(ABC):
    def __init__(self, base_handler_type: type, can_fit: bool, can_partial_fit: bool):
        self.name = None
        self.handler_type = None
        self.can_fit = can_fit
        self.can_partial_fit = can_partial_fit
        self.base_handler_type = base_handler_type

    @abstractmethod
    def create_handler(self) -> CustomHandler:
        ...

    def post_init(self, handler_type: type):
        self.handler_type = handler_type

        if self.can_be_used(handler_type):
            self.name = self.handler_type.__name__

    def can_be_used(self, handler_type: type) -> bool:
        return isinstance(handler_type, type) and inspect.isclass(handler_type) and issubclass(handler_type, self.base_handler_type)


# Implemented factories

class PerSeriesCustomHandlerFactory(CustomHandlerFactory):
    def __init__(self):
        super().__init__(PerSeriesCustomHandler, True, False)

    def create_handler(self) -> PerSeriesCustomHandler:
        return self.handler_type()


class AllSeriesCustomHandlerFactory(CustomHandlerFactory):
    def __init__(self):
        super().__init__(AllSeriesCustomHandler, True, False)

    def create_handler(self) -> AllSeriesCustomHandler:
        return self.handler_type()


class NoFitCustomHandlerFactory(CustomHandlerFactory):
    def __init__(self):
        super().__init__(NoFitCustomHandler, False, False)

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
