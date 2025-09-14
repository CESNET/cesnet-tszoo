from abc import ABC, abstractmethod
import warnings
import inspect

import numpy as np

from cesnet_tszoo.utils.enums import AnomalyHandlerType
from cesnet_tszoo.utils.constants import Z_SCORE, INTERQUARTILE_RANGE


class AnomalyHandler(ABC):
    """
    Base class for anomaly handlers, used for handling anomalies in the data.

    This class serves as the foundation for creating custom anomaly handlers. To implement a custom anomaly handler, this class is recommended to be subclassed and extended.

    Example:

        import numpy as np

        class InterquartileRange(AnomalyHandler):

            def __init__(self):
                self.lower_bound = None
                self.upper_bound = None
                self.iqr = None

            def fit(self, data: np.ndarray) -> None:
                q25, q75 = np.percentile(data, [25, 75], axis=0)
                self.iqr = q75 - q25

                self.lower_bound = q25 - 1.5 * self.iqr
                self.upper_bound = q75 + 1.5 * self.iqr

            def transform_anomalies(self, data: np.ndarray) -> np.ndarray:
                mask_lower_outliers = data < self.lower_bound
                mask_upper_outliers = data > self.upper_bound

                data[mask_lower_outliers] = np.take(self.lower_bound, np.where(mask_lower_outliers)[1])
                data[mask_upper_outliers] = np.take(self.upper_bound, np.where(mask_upper_outliers)[1])

    """

    IDENTIFIER = None

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Sets the anomaly handler values for a given time series part.

        This method must be implemented.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.  
        """
        ...

    @abstractmethod
    def transform_anomalies(self, data: np.ndarray):
        """
        Transforms anomalies the input data for a given time series part.

        This method must be implemented.
        Anomaly transformation is done in-place.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.            
        """
        ...


class ZScore(AnomalyHandler):
    """
    Fitting calculates mean and standard deviation of values used for fitting. 
    Calculated mean and standard deviation calculated when fitting will be used for calculating z-score for every value and those with z-score over or below threshold (3) will be clipped to the threshold value.

    Corresponds to enum [`AnomalyHandlerType.Z_SCORE`][cesnet_tszoo.utils.enums.AnomalyHandlerType] or literal `z-score`.
    """

    IDENTIFIER = "z-score"

    def __init__(self):
        self.mean = None
        self.std = None
        self.threshold = 3

    def fit(self, data: np.ndarray) -> None:
        warnings.filterwarnings("ignore")
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        warnings.filterwarnings("always")

    def transform_anomalies(self, data: np.ndarray):
        temp = data - self.mean
        z_score = np.divide(temp, self.std, out=np.zeros_like(temp, dtype=float), where=self.std != 0)
        mask_outliers = np.abs(z_score) > self.threshold

        clipped_values = self.mean + np.sign(z_score) * self.threshold * self.std

        data[mask_outliers] = clipped_values[mask_outliers]


class InterquartileRange(AnomalyHandler):
    """
    Fitting calculates 25th percentile, 75th percentile from the values used for fitting. From those percentiles the interquartile range, lower and upper bound will be calculated.
    Lower and upper bounds will then be used for detecting anomalies (values below lower bound or above upper bound). Anomalies will then be clipped to closest bound.

    Corresponds to enum [`AnomalyHandlerType.INTERQUARTILE_RANGE`][cesnet_tszoo.utils.enums.AnomalyHandlerType] or literal `interquartile_range`.
    """

    IDENTIFIER = "interquartile_range"

    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None
        self.iqr = None

    def fit(self, data: np.ndarray) -> None:
        q25, q75 = np.percentile(data, [25, 75], axis=0)
        self.iqr = q75 - q25

        self.lower_bound = q25 - 1.5 * self.iqr
        self.upper_bound = q75 + 1.5 * self.iqr

    def transform_anomalies(self, data: np.ndarray):
        mask_lower_outliers = data < self.lower_bound
        mask_upper_outliers = data > self.upper_bound

        data[mask_lower_outliers] = np.take(self.lower_bound, np.where(mask_lower_outliers)[1])
        data[mask_upper_outliers] = np.take(self.upper_bound, np.where(mask_upper_outliers)[1])


class NoAnomalyHandler(AnomalyHandler):
    """
    Does nothing. 

    Corresponds to enum [`AnomalyHandlerType.NO_ANOMALY_HANDLER`][cesnet_tszoo.utils.enums.AnomalyHandlerType] or literal `no_anomaly_handler`.
    """

    IDENTIFIER = AnomalyHandlerType.NO_ANOMALY_HANDLER.value

    def fit(self, data: np.ndarray) -> None:
        ...

    def transform_anomalies(self, data: np.ndarray):
        ...


class AnomalyHandlerFactory(ABC):
    """Base class for anomaly handler factories. """

    def __init__(self, anomaly_handler_type: type, creates_built_in: bool = True):
        self.anomaly_handler_type = anomaly_handler_type
        self.creates_built_in = creates_built_in

    @abstractmethod
    def create_anomaly_handler(self) -> AnomalyHandler:
        """Creates anomaly handler instance. """
        ...

    def can_be_used(self, handle_anomalies_with: AnomalyHandlerType | type) -> bool:
        """Checks whether factory can be used for passed anomaly handler. """

        if isinstance(handle_anomalies_with, AnomalyHandlerType):
            return handle_anomalies_with.value == self.anomaly_handler_type.IDENTIFIER

        return self.anomaly_handler_type == handle_anomalies_with


class ZScoreFactory(AnomalyHandlerFactory):
    """Factory class for ZScore anomaly handler. """

    def __init__(self):
        super().__init__(ZScore)

    def create_anomaly_handler(self) -> ZScore:
        return ZScore()


class InterquartileRangeFactory(AnomalyHandlerFactory):
    """Factory class for InterquartileRange anomaly handler. """

    def __init__(self):
        super().__init__(InterquartileRange)

    def create_anomaly_handler(self) -> InterquartileRange:
        return InterquartileRange()


class NoAnomalyHandlerFactory(AnomalyHandlerFactory):
    """Factory class for NoAnomalyHandler anomaly handler. """

    def __init__(self):
        super().__init__(NoAnomalyHandler)

    def create_anomaly_handler(self) -> NoAnomalyHandler:
        return NoAnomalyHandler()


class CustomAnomalyHandlerFactory(AnomalyHandlerFactory):
    """Factory class for custom anomaly handler. """

    def __init__(self, anomaly_handler_type: type):
        super().__init__(anomaly_handler_type, creates_built_in=False)

        if self.can_be_used(anomaly_handler_type) and anomaly_handler_type.IDENTIFIER is None:
            anomaly_handler_type.IDENTIFIER = f"{self.anomaly_handler_type.__name__} (Custom)"

    def create_anomaly_handler(self) -> AnomalyHandler:
        return self.anomaly_handler_type()

    def can_be_used(self, handle_anomalies_with: AnomalyHandlerType | type) -> bool:
        return inspect.isclass(handle_anomalies_with) and issubclass(handle_anomalies_with, AnomalyHandler)


def get_anomaly_handler_factory(handle_anomalies_with: AnomalyHandlerType | str | type | None) -> AnomalyHandlerFactory:
    """Creates anomaly handler factory for used anomaly handler. """

    # Validate and process anomaly handler type
    if isinstance(handle_anomalies_with, (str, AnomalyHandlerType)):
        handle_anomalies_with = AnomalyHandlerType(handle_anomalies_with)

    if handle_anomalies_with is None:
        handle_anomalies_with = AnomalyHandlerType.NO_ANOMALY_HANDLER

    anomaly_handler_factories = [NoAnomalyHandlerFactory(), ZScoreFactory(), InterquartileRangeFactory(), CustomAnomalyHandlerFactory(handle_anomalies_with)]
    for factory in anomaly_handler_factories:
        if factory.can_be_used(handle_anomalies_with):
            return factory

    raise TypeError("Passed anomaly handler type cannot be used! Either use built-in anomaly handlers or pass a custom anomaly handler that subclasses from AnomalyHandler base class.")
