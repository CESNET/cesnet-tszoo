from abc import ABC, abstractmethod
import warnings

from cesnet_tszoo.utils.constants import BASE_DATA_DTYPE_PART

import numpy as np


class AnomalyHandler(ABC):
    """
    Base class for anomaly handlers, used for handling anomalies in the data.

    This class serves as the foundation for creating custom anomaly handlers. To implement a custom anomaly handler, this class is recommended to be subclassed and extended.

    Example:

        import numpy as np

        class InterquartileRange(AnomalyHandler):

            def __init__(self):
                self.lower_bound = {}
                self.upper_bound = {}

            def fit(self, data: np.ndarray) -> None:

                warnings.filterwarnings("ignore")

                for name in data.dtype.names:
                    current_data = data[name]

                    q25, q75 = np.nanpercentile(current_data, [25, 75], axis=0)
                    iqr = q75 - q25

                    self.lower_bound[name] = q25 - 1.5 * iqr
                    self.upper_bound[name] = q75 + 1.5 * iqr

                warnings.filterwarnings("always")

            def transform_anomalies(self, data: np.ndarray):

                for name in data.dtype.names:
                    lower_bound = self.lower_bound[name]
                    upper_bound = self.upper_bound[name]
                    current_data = data[name]

                    lb_broadcast = np.broadcast_to(lower_bound, current_data.shape)
                    ub_broadcast = np.broadcast_to(upper_bound, current_data.shape)

                    mask_lower = current_data < lb_broadcast
                    mask_upper = current_data > ub_broadcast

                    current_data[mask_lower] = lb_broadcast[mask_lower]
                    current_data[mask_upper] = ub_broadcast[mask_upper]

    """

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Sets the anomaly handler values for a given time series part.

        This method must be implemented.

        Parameters:
            data: A structured numpy array representing data for a single time series with shape `(times)`. Use data["base_data"] to get non matrix features excluding any identifiers. 
                  For matrix features use their name instead of base_data.   
        """
        ...

    @abstractmethod
    def transform_anomalies(self, data: np.ndarray):
        """
        Transforms anomalies the input data for a given time series part.

        This method must be implemented.
        Anomaly transformation is done in-place.

        Parameters:
            data: A structured numpy array representing data for a single time series with shape `(times)`. Use data["base_data"] to get non matrix features excluding any identifiers. 
                  For matrix features use their name instead of base_data.   

        Returns:
            The changed data, with the same shape and dtype as the input `(times)`.                            
        """
        ...


class ZScore(AnomalyHandler):
    """
    Fitting calculates mean and standard deviation of values used for fitting. 
    Calculated mean and standard deviation calculated when fitting will be used for calculating z-score for every value and those with z-score over or below threshold (3) will be clipped to the threshold value.

    Corresponds to enum [`AnomalyHandlerType.Z_SCORE`][cesnet_tszoo.utils.enums.AnomalyHandlerType] or literal `z-score`.
    """

    def __init__(self):
        self.mean = {}
        self.std = {}
        self.threshold = 3

    def fit(self, data: np.ndarray) -> None:

        warnings.filterwarnings("ignore")

        for name in data.dtype.names:
            self.mean[name] = np.nanmean(data[name], axis=0)
            self.std[name] = np.nanstd(data[name], axis=0)

        warnings.filterwarnings("always")

    def transform_anomalies(self, data: np.ndarray):

        for name in data.dtype.names:

            mean = self.mean[name]
            std = self.std[name]
            current_data = data[name].view()

            temp = current_data - mean
            z_score = np.divide(temp, std, out=np.zeros_like(temp, dtype=float), where=std != 0)
            mask_outliers = np.abs(z_score) > self.threshold

            clipped_values = mean + np.sign(z_score) * self.threshold * std

            current_data[mask_outliers] = clipped_values[mask_outliers]


class InterquartileRange(AnomalyHandler):
    """
    Fitting calculates 25th percentile, 75th percentile from the values used for fitting. From those percentiles the interquartile range, lower and upper bound will be calculated.
    Lower and upper bounds will then be used for detecting anomalies (values below lower bound or above upper bound). Anomalies will then be clipped to closest bound.

    Corresponds to enum [`AnomalyHandlerType.INTERQUARTILE_RANGE`][cesnet_tszoo.utils.enums.AnomalyHandlerType] or literal `interquartile_range`.
    """

    def __init__(self):
        self.lower_bound = {}
        self.upper_bound = {}

    def fit(self, data: np.ndarray) -> None:

        warnings.filterwarnings("ignore")

        for name in data.dtype.names:
            current_data = data[name]

            q25, q75 = np.nanpercentile(current_data, [25, 75], axis=0)
            iqr = q75 - q25

            self.lower_bound[name] = q25 - 1.5 * iqr
            self.upper_bound[name] = q75 + 1.5 * iqr

        warnings.filterwarnings("always")

    def transform_anomalies(self, data: np.ndarray):

        for name in data.dtype.names:
            lower_bound = self.lower_bound[name]
            upper_bound = self.upper_bound[name]
            current_data = data[name]

            lb_broadcast = np.broadcast_to(lower_bound, current_data.shape)
            ub_broadcast = np.broadcast_to(upper_bound, current_data.shape)

            mask_lower = current_data < lb_broadcast
            mask_upper = current_data > ub_broadcast

            current_data[mask_lower] = lb_broadcast[mask_lower]
            current_data[mask_upper] = ub_broadcast[mask_upper]


class NoAnomalyHandler(AnomalyHandler):
    """
    Does nothing. 

    Corresponds to enum [`AnomalyHandlerType.NO_ANOMALY_HANDLER`][cesnet_tszoo.utils.enums.AnomalyHandlerType] or literal `no_anomaly_handler`.
    """

    def fit(self, data: np.ndarray) -> None:
        ...

    def transform_anomalies(self, data: np.ndarray):
        ...
