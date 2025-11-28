from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from cesnet_tszoo.utils.enums import SplitType


class CustomHandler(ABC):
    """
    Base class for custom handlers. Should not be used. Subclass from PerSeriesCustomHandler, AllSeriesCustomHandler or NoFitCustomHandler.

    This class serves as the foundation for creating custom handlers.
    Custom handlers are used to allow custom preprocess to be applied on data and specific set.
    """

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Applies on the input data for a given time series part.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.  

        Returns:
            The changed data, with the same shape as the input `(times, features)`.            
        """
        ...

    @staticmethod
    @abstractmethod
    def get_target_sets() -> set[SplitType] | set[Literal["train", "val", "test", "all"]]:
        """Specifies on which sets this handler should be used. """
        ...


class PerSeriesCustomHandler(CustomHandler):
    """
    Base class for PerSeriesCustomHandler. Used for custom handlers that are fitted on single train time series and then applied to that time series parts from target sets.

    This class serves as the foundation for creating PerSeriesCustomHandler handlers and should be subclassed from.

    Example:

        import numpy as np

        class PerFitTest(PerSeriesCustomHandler):

            def __init__(self):
                self.count = 0
                super().__init__()

            def fit(self, data: np.ndarray) -> None:
                self.count += 1

            def apply(self, data: np.ndarray) -> np.ndarray:
                data[:, :] = self.count
                return data

            @staticmethod
            def get_target_sets():
                return ["val"]             
    """

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Sets the PerSeriesCustomHandler values for a given time series data. Usually train set part of the time series.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.  
        """
        ...

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Applies on the input data for a given time series part.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.  

        Returns:
            The changed data, with the same shape as the input `(times, features)`.            
        """
        ...

    @staticmethod
    @abstractmethod
    def get_target_sets() -> set[SplitType] | set[Literal["train", "val", "test", "all"]]:
        """Specifies on which sets this handler should be used. """
        ...


class AllSeriesCustomHandler(CustomHandler):
    """
    Base class for AllSeriesCustomHandler. Used for custom handlers that are fitted on all train time series and then applied to all (from target sets) time series.

    This class serves as the foundation for creating AllSeriesCustomHandler handlers and should be subclassed from.

    Example:

        import numpy as np

        class AllFitTest(AllSeriesCustomHandler):

            def __init__(self):
                self.count = 0
                super().__init__()

            def partial_fit(self, data: np.ndarray) -> None:
                self.count += 1

            def apply(self, data: np.ndarray) -> np.ndarray:
                data[:, :] = self.count
                return data

            @staticmethod
            def get_target_sets():
                return ["train"]

    """

    @abstractmethod
    def partial_fit(self, data: np.ndarray) -> None:
        """
        Sets the AllSeriesCustomHandler values for a given time series data. Usually train set part of some time series.

        Parameters:
            data: A numpy array representing data for a time series with shape `(times, features)` excluding any identifiers.  
        """
        ...

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Applies on the input data for a given time series part.

        Parameters:
            data: A numpy array representing data for a time series with shape `(times, features)` excluding any identifiers.  

        Returns:
            The changed data, with the same shape as the input `(times, features)`.            
        """
        ...

    @staticmethod
    @abstractmethod
    def get_target_sets() -> set[SplitType] | set[Literal["train", "val", "test", "all"]]:
        """Specifies on which sets this handler should be used. """
        ...


class NoFitCustomHandler(CustomHandler):
    """
    Base class for NoFitCustomHandler. Used for custom handlers that are not fitted and are applied to (from target sets) time series.

    This class serves as the foundation for creating NoFitCustomHandler handlers and should be subclassed from.

    Example:

        import numpy as np

        class NoFitTest(NoFitCustomHandler):
            def apply(self, data: np.ndarray) -> np.ndarray:
                data[:, :] = -1
                return data

            @staticmethod
            def get_target_sets():
                return ["test"]

    """

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Applies on the input data for a given time series part.

        Parameters:
            data: A numpy array representing data for a time series with shape `(times, features)` excluding any identifiers.  

        Returns:
            The changed data, with the same shape as the input `(times, features)`.            
        """
        ...

    @staticmethod
    @abstractmethod
    def get_target_sets() -> set[SplitType] | set[Literal["train", "val", "test", "all"]]:
        """Specifies on which sets this handler should be used. """
        ...
