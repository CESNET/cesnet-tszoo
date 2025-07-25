from abc import ABC, abstractmethod

import numpy as np
import sklearn.preprocessing as sk

from cesnet_tszoo.utils.enums import ScalerType
from cesnet_tszoo.utils.constants import LOG_SCALER, L2_NORMALIZER, STANDARD_SCALER, MIN_MAX_SCALER, MAX_ABS_SCALER, POWER_TRANSFORMER, QUANTILE_TRANSFORMER, ROBUST_SCALER


class Scaler(ABC):
    """
    Base class for scalers, used for transforming data.

    This class serves as the foundation for creating custom scalers. To implement a custom scaler, this class is recommended to be subclassed and extended.

    Example:

        import numpy as np

        class LogScaler(Scaler):

            def fit(self, data: np.ndarray):
                ...

            def partial_fit(self, data: np.ndarray) -> None:
                ...

            def transform(self, data: np.ndarray):
                log_data = np.ma.log(data)

                return log_data.filled(np.nan)
    """

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Sets the scaler values for a given time series part.

        This method must be implemented if using multiple scalers that have not been pre-fitted.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.  
        """
        ...

    @abstractmethod
    def partial_fit(self, data: np.ndarray) -> None:
        """
        Partially sets the scaler values for a given time series part.

        This method must be implemented if using a single scaler that is not pre-fitted for all time series, or when using pre-fitted scaler(s) with `partial_fit_initialized_scalers` set to `True`.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.        
        """
        ...

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the input data for a given time series part.

        This method must always be implemented.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.  

        Returns:
            The transformed data, with the same shape as the input `(times, features)`.            
        """
        ...


class MinMaxScaler(Scaler):
    """
    Tranforms data using Scikit [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

    Corresponds to enum [`ScalerType.MIN_MAX_SCALER`][cesnet_tszoo.utils.enums.ScalerType] or literal `min_max_scaler`.
    """

    def __init__(self):
        self.scaler = sk.MinMaxScaler()

    def fit(self, data: np.ndarray):
        self.scaler.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.scaler.partial_fit(data)

    def transform(self, data: np.ndarray):
        return self.scaler.transform(data)


class StandardScaler(Scaler):
    """
    Tranforms data using Scikit [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

    Corresponds to enum [`ScalerType.STANDARD_SCALER`][cesnet_tszoo.utils.enums.ScalerType] or literal `standard_scaler`.
    """

    def __init__(self):
        self.scaler = sk.StandardScaler()

    def fit(self, data: np.ndarray):
        self.scaler.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.scaler.partial_fit(data)

    def transform(self, data: np.ndarray):
        return self.scaler.transform(data)


class MaxAbsScaler(Scaler):
    """
    Tranforms data using Scikit [`MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).

    Corresponds to enum [`ScalerType.MAX_ABS_SCALER`][cesnet_tszoo.utils.enums.ScalerType] or literal `max_abs_scaler`.
    """

    def __init__(self):
        self.scaler = sk.MaxAbsScaler()

    def fit(self, data: np.ndarray):
        self.scaler.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.scaler.partial_fit(data)

    def transform(self, data: np.ndarray):
        return self.scaler.transform(data)


class LogScaler(Scaler):
    """
    Tranforms data with natural logarithm. Zero or invalid values are set to `np.nan`.

    Corresponds to enum [`ScalerType.LOG_SCALER`][cesnet_tszoo.utils.enums.ScalerType] or literal `log_scaler`.
    """

    def fit(self, data: np.ndarray):
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray):
        log_data = np.ma.log(data)

        return log_data.filled(np.nan)


class L2Normalizer(Scaler):
    """
    Tranforms data using Scikit [`L2Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html).

    Corresponds to enum [`ScalerType.L2_NORMALIZER`][cesnet_tszoo.utils.enums.ScalerType] or literal `l2_normalizer`.
    """

    def __init__(self):
        self.scaler = sk.Normalizer(norm="l2")

    def fit(self, data: np.ndarray):
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray):
        return self.scaler.fit_transform(data)


class RobustScaler(Scaler):
    """
    Tranforms data using Scikit [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

    Corresponds to enum [`ScalerType.ROBUST_SCALER`][cesnet_tszoo.utils.enums.ScalerType] or literal `robust_scaler`.

    !!! warning "partial_fit not supported"
        Because this scaler does not support partial_fit it can't be used when using one scaler that needs to be fitted for multiple time series.    
    """

    def __init__(self):
        self.scaler = sk.RobustScaler()

    def fit(self, data: np.ndarray):
        self.scaler.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("RobustScaler does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.scaler.transform(data)


class PowerTransformer(Scaler):
    """
    Tranforms data using Scikit [`PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html).

    Corresponds to enum [`ScalerType.POWER_TRANSFORMER`][cesnet_tszoo.utils.enums.ScalerType] or literal `power_transformer`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one scaler that needs to be fitted for multiple time series.
    """

    def __init__(self):
        self.scaler = sk.PowerTransformer()

    def fit(self, data: np.ndarray):
        self.scaler.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("PowerTransformer does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.scaler.transform(data)


class QuantileTransformer(Scaler):
    """
    Tranforms data using Scikit [`QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html).

    Corresponds to enum [`ScalerType.QUANTILE_TRANSFORMER`][cesnet_tszoo.utils.enums.ScalerType] or literal `quantile_transformer`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one scaler that needs to be fitted for multiple time series.    
    """

    def __init__(self):
        self.scaler = sk.QuantileTransformer()

    def fit(self, data: np.ndarray):
        self.scaler.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("QuantileTransformer does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.scaler.transform(data)


def input_has_fit_method(to_check) -> bool:
    """Checks whether `to_check` has fit method. """

    fit_method = getattr(to_check, "fit", None)
    if callable(fit_method):
        return True

    return False


def input_has_partial_fit_method(to_check) -> bool:
    """Checks whether `to_check` has partial_fit method. """

    partial_fit_method = getattr(to_check, "partial_fit", None)
    if callable(partial_fit_method):
        return True

    return False


def input_has_transform(to_check) -> bool:
    """Checks whether `to_check` has transform method. """

    transform_method = getattr(to_check, "transform", None)
    if callable(transform_method):
        return True

    return False


def scaler_from_input_to_scaler_type(scaler_from_input: ScalerType | type, check_for_fit: bool, check_for_partial_fit: bool) -> tuple[type, str]:
    """Converts from input to type value and str that represents scaler's name."""

    if scaler_from_input is None:
        return None, None

    if scaler_from_input == StandardScaler or scaler_from_input == ScalerType.STANDARD_SCALER:
        return StandardScaler, STANDARD_SCALER
    elif scaler_from_input == L2Normalizer or scaler_from_input == ScalerType.L2_NORMALIZER:
        return L2Normalizer, L2_NORMALIZER
    elif scaler_from_input == LogScaler or scaler_from_input == ScalerType.LOG_SCALER:
        return LogScaler, LOG_SCALER
    elif scaler_from_input == MaxAbsScaler or scaler_from_input == ScalerType.MAX_ABS_SCALER:
        return MaxAbsScaler, MAX_ABS_SCALER
    elif scaler_from_input == MinMaxScaler or scaler_from_input == ScalerType.MIN_MAX_SCALER:
        return MinMaxScaler, MIN_MAX_SCALER
    elif scaler_from_input == PowerTransformer or scaler_from_input == ScalerType.POWER_TRANSFORMER:
        return PowerTransformer, POWER_TRANSFORMER
    elif scaler_from_input == QuantileTransformer or scaler_from_input == ScalerType.QUANTILE_TRANSFORMER:
        return QuantileTransformer, QUANTILE_TRANSFORMER
    elif scaler_from_input == RobustScaler or scaler_from_input == ScalerType.ROBUST_SCALER:
        return RobustScaler, ROBUST_SCALER
    else:

        assert input_has_transform(scaler_from_input)
        if check_for_fit:
            assert input_has_fit_method(scaler_from_input)

        if check_for_partial_fit:
            assert input_has_partial_fit_method(scaler_from_input)

        return scaler_from_input, f"{scaler_from_input.__name__} (Custom)"
