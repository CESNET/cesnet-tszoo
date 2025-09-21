from abc import ABC, abstractmethod
import inspect

from cesnet_tszoo.utils.enums import AnomalyHandlerType
from cesnet_tszoo.utils.anomaly_handler.anomaly_handler import AnomalyHandler, ZScore, InterquartileRange, NoAnomalyHandler


class AnomalyHandlerFactory(ABC):
    """Base class for anomaly handler factories. """

    def __init__(self, anomaly_handler_type: type, identifier: AnomalyHandlerType | None, creates_built_in: bool = True, is_empty_factory: bool = False):
        self.anomaly_handler_type = anomaly_handler_type
        self.creates_built_in = creates_built_in
        self.identifier = identifier
        self.is_empty_factory = is_empty_factory

        if isinstance(anomaly_handler_type, type):
            self.name = anomaly_handler_type.__name__

    def post_init(self, anomaly_handler_type: type):
        """Called after has constructor passed from outside. """
        ...

    @abstractmethod
    def create_anomaly_handler(self) -> AnomalyHandler:
        """Creates anomaly handler instance. """
        ...

    def can_be_used(self, handle_anomalies_with: AnomalyHandlerType | type) -> bool:
        """Checks whether factory can be used for passed anomaly handler. """

        if isinstance(handle_anomalies_with, AnomalyHandlerType):
            return handle_anomalies_with == self.identifier

        return self.anomaly_handler_type == handle_anomalies_with


# Implemented factories

class ZScoreFactory(AnomalyHandlerFactory):
    """Factory class for ZScore anomaly handler. """

    def __init__(self):
        super().__init__(ZScore, AnomalyHandlerType.Z_SCORE)

    def create_anomaly_handler(self) -> ZScore:
        return ZScore()


class InterquartileRangeFactory(AnomalyHandlerFactory):
    """Factory class for InterquartileRange anomaly handler. """

    def __init__(self):
        super().__init__(InterquartileRange, AnomalyHandlerType.INTERQUARTILE_RANGE)

    def create_anomaly_handler(self) -> InterquartileRange:
        return InterquartileRange()


class NoAnomalyHandlerFactory(AnomalyHandlerFactory):
    """Factory class for NoAnomalyHandler anomaly handler. """

    def __init__(self):
        super().__init__(NoAnomalyHandler, AnomalyHandlerType.NO_ANOMALY_HANDLER, is_empty_factory=True)

    def create_anomaly_handler(self) -> NoAnomalyHandler:
        return NoAnomalyHandler()


class CustomAnomalyHandlerFactory(AnomalyHandlerFactory):
    """Factory class for custom anomaly handler. """

    def __init__(self):
        super().__init__(None, None, creates_built_in=False)

    def create_anomaly_handler(self) -> AnomalyHandler:
        return self.anomaly_handler_type()

    def can_be_used(self, handle_anomalies_with: type) -> bool:
        return isinstance(handle_anomalies_with, type) and inspect.isclass(handle_anomalies_with) and issubclass(handle_anomalies_with, AnomalyHandler)

    def post_init(self, anomaly_handler_type: type):
        self.anomaly_handler_type = anomaly_handler_type

        if self.can_be_used(anomaly_handler_type):
            self.name = f"{self.anomaly_handler_type.__name__} (Custom)"

# Implemented factories


def get_anomaly_handler_factory(handle_anomalies_with: AnomalyHandlerType | str | type | None) -> AnomalyHandlerFactory:
    """Creates anomaly handler factory for used anomaly handler. """

    # Validate and process anomaly handler type
    if isinstance(handle_anomalies_with, (str, AnomalyHandlerType)):
        handle_anomalies_with = AnomalyHandlerType(handle_anomalies_with)

    if handle_anomalies_with is None:
        handle_anomalies_with = AnomalyHandlerType.NO_ANOMALY_HANDLER

    for factory in AnomalyHandlerFactory.__subclasses__():
        factory_instance = factory()
        factory_instance.post_init(handle_anomalies_with)

        if factory_instance.can_be_used(handle_anomalies_with):
            return factory_instance

    raise TypeError("Passed anomaly handler type cannot be used! Either use built-in anomaly handlers or pass a custom anomaly handler that subclasses from AnomalyHandler base class.")
