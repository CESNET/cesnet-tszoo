from abc import ABC, abstractmethod
import warnings

import numpy as np

from cesnet_tszoo.utils.enums import AnomalyHandlerType
from cesnet_tszoo.utils.constants import Z_SCORE, INTERQUARTILE_RANGE


class AnomalyHandler(ABC):

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        ...

    @abstractmethod
    def transform_anomalies(self, data: np.ndarray, default_values: np.ndarray) -> np.ndarray:
        ...


class ZScore(AnomalyHandler):

    def __init__(self):
        self.mean = None
        self.std = None
        self.threshold = 3

    def fit(self, data: np.ndarray):
        warnings.filterwarnings("ignore")
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        warnings.filterwarnings("always")

    def transform_anomalies(self, data: np.ndarray, default_values: np.ndarray):
        temp = data - self.mean
        z_score = np.divide(temp, self.std, out=np.zeros_like(temp, dtype=float), where=self.std != 0)
        mask_outliers = np.abs(z_score) > self.threshold

        clipped_values = self.mean + np.sign(z_score) * self.threshold * self.std

        data[mask_outliers] = clipped_values[mask_outliers]


class InterquartileRange(AnomalyHandler):

    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None
        self.iqr = None

    def fit(self, data: np.ndarray):
        q25, q75 = np.percentile(data, [25, 75], axis=0)
        self.iqr = q75 - q25

        self.lower_bound = q25 - 1.5 * self.iqr
        self.upper_bound = q75 + 1.5 * self.iqr

    def transform_anomalies(self, data: np.ndarray, default_values: np.ndarray):
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
    """Checks whether `to_check` has transform method. """

    transform_method = getattr(to_check, "transform", None)
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
