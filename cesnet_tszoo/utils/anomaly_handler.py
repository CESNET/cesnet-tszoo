from abc import ABC, abstractmethod
import warnings

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
    def transform_anomalies(self, data: np.ndarray) -> np.ndarray:
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

    def __init__(self):
        self.mean = None
        self.std = None
        self.threshold = 3

    def fit(self, data: np.ndarray) -> None:
        warnings.filterwarnings("ignore")
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        warnings.filterwarnings("always")

    def transform_anomalies(self, data: np.ndarray) -> np.ndarray:
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


def input_has_fit_method(to_check) -> bool:
    """Checks whether `to_check` has fit method. """

    fit_method = getattr(to_check, "fit", None)
    if callable(fit_method):
        return True

    return False


def input_has_transform(to_check) -> bool:
    """Checks whether `to_check` has transform_anomalies method. """

    transform_method = getattr(to_check, "transform_anomalies", None)
    if callable(transform_method):
        return True

    return False


def anomaly_handler_from_input_to_anomaly_handler_type(anomaly_handler_from_input: AnomalyHandlerType | type) -> tuple[type, str]:

    if anomaly_handler_from_input is None:
        return None, None

    if anomaly_handler_from_input == ZScore or anomaly_handler_from_input == AnomalyHandlerType.Z_SCORE:
        return ZScore, Z_SCORE
    elif anomaly_handler_from_input == InterquartileRange or anomaly_handler_from_input == AnomalyHandlerType.INTERQUARTILE_RANGE:
        return InterquartileRange, INTERQUARTILE_RANGE
    else:

        assert input_has_transform(anomaly_handler_from_input)
        assert input_has_fit_method(anomaly_handler_from_input)

        return anomaly_handler_from_input, f"{anomaly_handler_from_input.__name__} (Custom)"
