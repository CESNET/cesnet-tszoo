from abc import ABC, abstractmethod
import inspect

from cesnet_tszoo.utils.enums import FillerType
from cesnet_tszoo.utils.filler.filler import Filler, MeanFiller, ForwardFiller, LinearInterpolationFiller, NoFiller


class FillerFactory(ABC):
    """Base class for filler factories. """

    def __init__(self, filler_type: type, identifier: FillerType | None, creates_built_in: bool = True):
        self.filler_type = filler_type
        self.creates_built_in = creates_built_in
        self.identifier = identifier

        if isinstance(filler_type, type):
            self.name = filler_type.__name__

    @abstractmethod
    def create_filler(self, features) -> Filler:
        """Creates filler instance. """
        ...

    def can_be_used(self, fill_missing_with: FillerType | type | None) -> bool:
        """Checks whether factory can be used for passed filler. """

        if isinstance(fill_missing_with, FillerType):
            return fill_missing_with == self.identifier

        return self.filler_type == fill_missing_with


# Implemented factories

class MeanFillerFactory(FillerFactory):
    """Factory class for MeanFiller. """

    def __init__(self):
        super().__init__(MeanFiller, FillerType.MEAN_FILLER)

    def create_filler(self, features) -> MeanFiller:
        return MeanFiller(features)


class ForwardFillerFactory(FillerFactory):
    """Factory class for ForwardFiller. """

    def __init__(self):
        super().__init__(ForwardFiller, FillerType.FORWARD_FILLER)

    def create_filler(self, features) -> ForwardFiller:
        return ForwardFiller(features)


class LinearInterpolationFillerFactory(FillerFactory):
    """Factory class for LinearInterpolationFiller. """

    def __init__(self):
        super().__init__(LinearInterpolationFiller, FillerType.LINEAR_INTERPOLATION_FILLER)

    def create_filler(self, features) -> LinearInterpolationFiller:
        return LinearInterpolationFiller(features)


class NoFillerFactory(FillerFactory):
    """Factory class for NoFiller. """

    def __init__(self):
        super().__init__(NoFiller, FillerType.NO_FILLER)

    def create_filler(self, features) -> NoFiller:
        return NoFiller(features)


class CustomFillerFactory(FillerFactory):
    """Factory class for custom fillers. """

    def __init__(self, filler_type: type):
        super().__init__(filler_type, None, creates_built_in=False)

        if self.can_be_used(filler_type):
            self.name = f"{self.filler_type.__name__} (Custom)"

    def create_filler(self, features) -> Filler:
        return self.filler_type(features)

    def can_be_used(self, fill_missing_with: type):
        return isinstance(fill_missing_with, type) and inspect.isclass(fill_missing_with) and issubclass(fill_missing_with, Filler)

# Implemented factories


def get_filler_factory(fill_missing_with: FillerType | str | type | None) -> FillerFactory:
    """Creates filler factory for used filler. """

    # Validate and process missing data filler type
    if isinstance(fill_missing_with, (str, FillerType)):
        fill_missing_with = FillerType(fill_missing_with)

    if fill_missing_with is None:
        fill_missing_with = FillerType.NO_FILLER

    filler_factories = [NoFillerFactory(), MeanFillerFactory(), ForwardFillerFactory(), LinearInterpolationFillerFactory(), CustomFillerFactory(fill_missing_with)]
    for factory in filler_factories:
        if factory.can_be_used(fill_missing_with):
            return factory

    raise TypeError("Passed filler type cannot be used! Either use built-in fillers or pass a custom filler that subclasses from Filler base class.")
