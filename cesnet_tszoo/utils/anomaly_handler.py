from abc import ABC, abstractmethod
import warnings

import numpy as np
import sklearn.preprocessing as sk

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

    def fit(self, data: np.ndarray):
        warnings.filterwarnings("ignore")
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        warnings.filterwarnings("always")

    def transform_anomalies(self, data: np.ndarray, default_values: np.ndarray):
        temp = data - self.mean
        z_score = np.divide(temp, self.std, out=np.zeros_like(temp, dtype=float), where=self.std != 0)
        mask_outliers = np.abs(z_score) > 3

        data[mask_outliers] = np.take(default_values, np.where(mask_outliers)[1])


class InterquartileRange(AnomalyHandler):

    def __init__(self):
        self.transformer = sk.MinMaxScaler()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def transform_anomalies(self, data: np.ndarray, default_values: np.ndarray):
        return self.transformer.transform(data)


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
