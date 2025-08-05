from abc import ABC, abstractmethod

import numpy as np
import sklearn.preprocessing as sk

from cesnet_tszoo.utils.enums import TransformerType
from cesnet_tszoo.utils.constants import LOG_TRANSFORMER, L2_NORMALIZER, STANDARD_SCALER, MIN_MAX_SCALER, MAX_ABS_SCALER, POWER_TRANSFORMER, QUANTILE_TRANSFORMER, ROBUST_SCALER


class Transformer(ABC):
    """
    Base class for transformers, used for transforming data.

    This class serves as the foundation for creating custom transformers. To implement a custom transformer, this class is recommended to be subclassed and extended.

    Example:

        import numpy as np

        class LogTransformer(Transformer):

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
        Sets the transformer values for a given time series part.

        This method must be implemented if using multiple transformers that have not been pre-fitted.

        Parameters:
            data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.  
        """
        ...

    @abstractmethod
    def partial_fit(self, data: np.ndarray) -> None:
        """
        Partially sets the transformer values for a given time series part.

        This method must be implemented if using a single transformer that is not pre-fitted for all time series, or when using pre-fitted transformer(s) with `partial_fit_initialized_transformers` set to `True`.

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


class MinMaxScaler(Transformer):
    """
    Tranforms data using Scikit [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

    Corresponds to enum [`TransformerType.MIN_MAX_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `min_max_scaler`.
    """

    def __init__(self):
        self.transformer = sk.MinMaxScaler()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.transformer.partial_fit(data)

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)


class StandardScaler(Transformer):
    """
    Tranforms data using Scikit [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

    Corresponds to enum [`TransformerType.STANDARD_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `standard_scaler`.
    """

    def __init__(self):
        self.transformer = sk.StandardScaler()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.transformer.partial_fit(data)

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)


class MaxAbsScaler(Transformer):
    """
    Tranforms data using Scikit [`MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).

    Corresponds to enum [`TransformerType.MAX_ABS_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `max_abs_scaler`.
    """

    def __init__(self):
        self.transformer = sk.MaxAbsScaler()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.transformer.partial_fit(data)

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)


class LogTransformer(Transformer):
    """
    Tranforms data with natural logarithm. Zero or invalid values are set to `np.nan`.

    Corresponds to enum [`TransformerType.LOG_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `log_transformer`.
    """

    def fit(self, data: np.ndarray):
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray):
        log_data = np.ma.log(data)

        return log_data.filled(np.nan)


class L2Normalizer(Transformer):
    """
    Tranforms data using Scikit [`L2Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html).

    Corresponds to enum [`TransformerType.L2_NORMALIZER`][cesnet_tszoo.utils.enums.TransformerType] or literal `l2_normalizer`.
    """

    def __init__(self):
        self.transformer = sk.Normalizer(norm="l2")

    def fit(self, data: np.ndarray):
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray):
        return self.transformer.fit_transform(data)


class RobustScaler(Transformer):
    """
    Tranforms data using Scikit [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

    Corresponds to enum [`TransformerType.ROBUST_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `robust_scaler`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.    
    """

    def __init__(self):
        self.transformer = sk.RobustScaler()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("RobustScaler does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)


class PowerTransformer(Transformer):
    """
    Tranforms data using Scikit [`PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html).

    Corresponds to enum [`TransformerType.POWER_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `power_transformer`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.
    """

    def __init__(self):
        self.transformer = sk.PowerTransformer()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("PowerTransformer does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)


class QuantileTransformer(Transformer):
    """
    Tranforms data using Scikit [`QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html).

    Corresponds to enum [`TransformerType.QUANTILE_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `quantile_transformer`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.    
    """

    def __init__(self):
        self.transformer = sk.QuantileTransformer()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("QuantileTransformer does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)


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


def transformer_from_input_to_transformer_type(transformer_from_input: TransformerType | type, check_for_fit: bool, check_for_partial_fit: bool) -> tuple[type, str]:
    """Converts from input to type value and str that represents transformer's name."""

    if transformer_from_input is None:
        return None, None

    if transformer_from_input == StandardScaler or transformer_from_input == TransformerType.STANDARD_SCALER:
        return StandardScaler, STANDARD_SCALER
    elif transformer_from_input == L2Normalizer or transformer_from_input == TransformerType.L2_NORMALIZER:
        return L2Normalizer, L2_NORMALIZER
    elif transformer_from_input == LogTransformer or transformer_from_input == TransformerType.LOG_TRANSFORMER:
        return LogTransformer, LOG_TRANSFORMER
    elif transformer_from_input == MaxAbsScaler or transformer_from_input == TransformerType.MAX_ABS_SCALER:
        return MaxAbsScaler, MAX_ABS_SCALER
    elif transformer_from_input == MinMaxScaler or transformer_from_input == TransformerType.MIN_MAX_SCALER:
        return MinMaxScaler, MIN_MAX_SCALER
    elif transformer_from_input == PowerTransformer or transformer_from_input == TransformerType.POWER_TRANSFORMER:
        return PowerTransformer, POWER_TRANSFORMER
    elif transformer_from_input == QuantileTransformer or transformer_from_input == TransformerType.QUANTILE_TRANSFORMER:
        return QuantileTransformer, QUANTILE_TRANSFORMER
    elif transformer_from_input == RobustScaler or transformer_from_input == TransformerType.ROBUST_SCALER:
        return RobustScaler, ROBUST_SCALER
    else:

        assert input_has_transform(transformer_from_input)
        if check_for_fit:
            assert input_has_fit_method(transformer_from_input)

        if check_for_partial_fit:
            assert input_has_partial_fit_method(transformer_from_input)

        return transformer_from_input, f"{transformer_from_input.__name__} (Custom)"
