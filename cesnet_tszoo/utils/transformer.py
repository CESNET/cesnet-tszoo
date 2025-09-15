from abc import ABC, abstractmethod
import inspect

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

            def inverse_transform(self, transformed_data):
                return np.exp(transformed_data)                
    """

    IDENTIFIER = None

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

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """
        Transforms the input transformed data to their original representation for a given time series part.

        Parameters:
            transformed_data: A numpy array representing data for a single time series with shape `(times, features)` excluding any identifiers.  

        Returns:
            The original representation of transformed data, with the same shape as the input `(times, features)`.            
        """
        return transformed_data


class MinMaxScaler(Transformer):
    """
    Tranforms data using Scikit [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

    Corresponds to enum [`TransformerType.MIN_MAX_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `min_max_scaler`.
    """

    IDENTIFIER = TransformerType.MIN_MAX_SCALER.value

    def __init__(self):
        self.transformer = sk.MinMaxScaler()

    def fit(self, data: np.ndarray) -> None:
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.transformer.partial_fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.transformer.transform(data)

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        return self.transformer.inverse_transform(transformed_data)


class StandardScaler(Transformer):
    """
    Tranforms data using Scikit [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

    Corresponds to enum [`TransformerType.STANDARD_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `standard_scaler`.
    """

    IDENTIFIER = TransformerType.STANDARD_SCALER.value

    def __init__(self):
        self.transformer = sk.StandardScaler()

    def fit(self, data: np.ndarray) -> None:
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.transformer.partial_fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.transformer.transform(data)

    def inverse_transform(self, transformed_data: np.ndarray):
        return self.transformer.inverse_transform(transformed_data)


class MaxAbsScaler(Transformer):
    """
    Tranforms data using Scikit [`MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).

    Corresponds to enum [`TransformerType.MAX_ABS_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `max_abs_scaler`.
    """

    IDENTIFIER = TransformerType.MAX_ABS_SCALER.value

    def __init__(self):
        self.transformer = sk.MaxAbsScaler()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.transformer.partial_fit(data)

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)

    def inverse_transform(self, transformed_data: np.ndarray):
        return self.transformer.inverse_transform(transformed_data)


class LogTransformer(Transformer):
    """
    Tranforms data with natural logarithm. Zero or invalid values are set to `np.nan`.

    Corresponds to enum [`TransformerType.LOG_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `log_transformer`.
    """

    IDENTIFIER = TransformerType.LOG_TRANSFORMER.value

    def fit(self, data: np.ndarray):
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray):
        log_data = np.ma.log(data)

        return log_data.filled(np.nan)

    def inverse_transform(self, transformed_data: np.ndarray):
        return np.exp(transformed_data)


class L2Normalizer(Transformer):
    """
    Tranforms data using Scikit [`L2Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html).

    Corresponds to enum [`TransformerType.L2_NORMALIZER`][cesnet_tszoo.utils.enums.TransformerType] or literal `l2_normalizer`.
    """

    IDENTIFIER = TransformerType.L2_NORMALIZER.value

    def __init__(self):
        self.transformer = sk.Normalizer(norm="l2")

    def fit(self, data: np.ndarray):
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray):
        return self.transformer.fit_transform(data)

    def inverse_transform(self, transformed_data: np.ndarray):
        raise NotImplementedError("Normalizer does not support inverse_transform.")


class RobustScaler(Transformer):
    """
    Tranforms data using Scikit [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

    Corresponds to enum [`TransformerType.ROBUST_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `robust_scaler`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.    
    """

    IDENTIFIER = TransformerType.ROBUST_SCALER.value

    def __init__(self):
        self.transformer = sk.RobustScaler()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("RobustScaler does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)

    def inverse_transform(self, transformed_data: np.ndarray):
        return self.transformer.inverse_transform(transformed_data)


class PowerTransformer(Transformer):
    """
    Tranforms data using Scikit [`PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html).

    Corresponds to enum [`TransformerType.POWER_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `power_transformer`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.
    """

    IDENTIFIER = TransformerType.POWER_TRANSFORMER.value

    def __init__(self):
        self.transformer = sk.PowerTransformer()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("PowerTransformer does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)

    def inverse_transform(self, transformed_data: np.ndarray):
        return self.transformer.inverse_transform(transformed_data)


class QuantileTransformer(Transformer):
    """
    Tranforms data using Scikit [`QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html).

    Corresponds to enum [`TransformerType.QUANTILE_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `quantile_transformer`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.    
    """

    IDENTIFIER = TransformerType.QUANTILE_TRANSFORMER.value

    def __init__(self):
        self.transformer = sk.QuantileTransformer()

    def fit(self, data: np.ndarray):
        self.transformer.fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("QuantileTransformer does not support partial_fit.")

    def transform(self, data: np.ndarray):
        return self.transformer.transform(data)

    def inverse_transform(self, transformed_data: np.ndarray):
        return self.transformer.inverse_transform(transformed_data)


class NoTransformer(Transformer):
    """
    Does nothing.

    Corresponds to enum [`TransformerType.NO_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `no_transformer`.
    """

    IDENTIFIER = TransformerType.NO_TRANSFORMER.value

    def fit(self, data: np.ndarray):
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        return transformed_data


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
    """Checks wheter type has transform method. """

    transform_method = getattr(to_check, "transform", None)
    if callable(transform_method):
        return True

    return False


class TransformerFactory(ABC):
    """Base class for transformer factories. """

    def __init__(self, transformer_type: type, can_fit: bool, can_partial_fit: bool, creates_built_in: bool = True, is_empty_factory: bool = False):
        self.transformer_type = transformer_type
        self.can_fit = can_fit
        self.can_partial_fit = can_partial_fit
        self.creates_built_in = creates_built_in
        self.has_already_initialized = False
        self.initialized_transformers = None
        self.is_empty_factory = is_empty_factory
        self.has_single_initialized = None

    @abstractmethod
    def create_transformer(self) -> Transformer:
        """Creates transformer instance. """
        ...

    def set_already_initialized_transformers(self, transform_with: Transformer | np.ndarray[Transformer] | list[Transformer]):
        """Sets already initalized transformers and set respective flag. """
        self.has_already_initialized = True

        if isinstance(transform_with, list):
            transform_with = np.array(transform_with)

        self.has_single_initialized = isinstance(transform_with, Transformer)

        self.initialized_transformers = transform_with

    def get_already_initialized_transformers(self) -> Transformer | np.ndarray[Transformer]:
        """Returns already initialized transformer/s if they exist. """
        assert self.has_already_initialized, "Factory must contain already initialized transformer/s."

        return self.initialized_transformers

    def can_be_used(self, transformer_type: TransformerType | type) -> bool:
        """Checks whether factory can be used for passed transformer. """

        if isinstance(transformer_type, TransformerType):
            return transformer_type.value == self.transformer_type.IDENTIFIER

        return isinstance(transformer_type, type) and self.transformer_type == transformer_type

    def raise_when_initialized_not_supported(self, create_transformer_per_time_series: bool, partial_fit_initialized_transformers: bool):
        """Validates whether initialized transformers can be used. """

        if not self.has_already_initialized:
            return

        if partial_fit_initialized_transformers:
            assert self.can_partial_fit, "Partial fit must be supported if you want to partial fit already initialized transformer/s. Or you can set partial_fit_initialized_transformers to False."

        if isinstance(self.initialized_transformers, np.ndarray):
            assert create_transformer_per_time_series, "When using multiple initialized transformers create_transformer_per_time_series must be set to True. Or use only one initialized transformer. "

        if create_transformer_per_time_series:
            assert isinstance(self.initialized_transformers, np.ndarray), "When create_transformer_per_time_series is set to True then you must use multiple initialized transformers for each used time series. Or set create_transformer_per_time_series to False."

    def raise_when_not_supported(self, create_transformer_per_time_series: bool):
        """Validates whether this transformer is supported. """

        assert input_has_transform(self.transformer_type), "Transformer must have method transform"

        if self.has_already_initialized:
            return

        if create_transformer_per_time_series:
            assert self.can_fit, "Fit must be supported if you want to create transformer per time series."

        if not create_transformer_per_time_series:
            assert self.can_partial_fit, "Partial fit must be supported if you want to use one transformer for multiple time series."


class StandardScalerFactory(TransformerFactory):
    """Factory class for StandardScaler. """

    def __init__(self):
        super().__init__(StandardScaler, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> StandardScaler:
        return StandardScaler()


class L2NormalizerFactory(TransformerFactory):
    """Factory class for L2Normalizer. """

    def __init__(self):
        super().__init__(L2Normalizer, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> L2Normalizer:
        return L2Normalizer()


class LogTransformerFactory(TransformerFactory):
    """Factory class for LogTransformer. """

    def __init__(self):
        super().__init__(LogTransformer, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> LogTransformer:
        return LogTransformer()


class MaxAbsScalerFactory(TransformerFactory):
    """Factory class for MaxAbsScaler. """

    def __init__(self):
        super().__init__(MaxAbsScaler, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> MaxAbsScaler:
        return MaxAbsScaler()


class MinMaxScalerFactory(TransformerFactory):
    """Factory class for MinMaxScaler. """

    def __init__(self):
        super().__init__(MinMaxScaler, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> MinMaxScaler:
        return MinMaxScaler()


class PowerTransformerFactory(TransformerFactory):
    """Factory class for PowerTransformer. """

    def __init__(self):
        super().__init__(PowerTransformer, can_fit=True, can_partial_fit=False)

    def create_transformer(self) -> PowerTransformer:
        return PowerTransformer()


class QuantileTransformerFactory(TransformerFactory):
    """Factory class for QuantileTransformer. """

    def __init__(self):
        super().__init__(QuantileTransformer, can_fit=True, can_partial_fit=False)

    def create_transformer(self) -> QuantileTransformer:
        return QuantileTransformer()


class RobustScalerFactory(TransformerFactory):
    """Factory class for RobustScaler. """

    def __init__(self):
        super().__init__(RobustScaler, can_fit=True, can_partial_fit=False)

    def create_transformer(self) -> RobustScaler:
        return RobustScaler()


class NoTransformerFactory(TransformerFactory):
    """Factory class for NoTransformer. """

    def __init__(self):
        super().__init__(NoTransformer, can_fit=True, can_partial_fit=True, is_empty_factory=True)

    def create_transformer(self) -> NoTransformer:
        return NoTransformer()


class CustomTransformerFactory(TransformerFactory):
    """Factory class for custom transformer. """

    def __init__(self, transform_with: type):
        super().__init__(transform_with, can_fit=None, can_partial_fit=None, creates_built_in=False)

        if self.can_be_used(transform_with):
            if not hasattr(transform_with, "IDENTIFIER") or transform_with.IDENTIFIER is None:
                transform_with.IDENTIFIER = f"{transform_with.__name__} (Custom)"

            self.can_fit = input_has_fit_method(transform_with)
            self.can_partial_fit = input_has_partial_fit_method(transform_with)

    def create_transformer(self) -> Transformer:
        return self.transformer_type()

    def can_be_used(self, trasformer_type: type) -> bool:
        return isinstance(trasformer_type, type) and inspect.isclass(trasformer_type)


def get_transformer_type_or_enum_and_validate(transform_with: TransformerType | str | type | None) -> TransformerType | type:
    """Returns type or enum variant of the passed transform_with. """

    if transform_with is None:
        return TransformerType.NO_TRANSFORMER

    if isinstance(transform_with, (str, TransformerType)):
        return TransformerType(transform_with)

    if isinstance(transform_with, type):
        return transform_with

    if isinstance(transform_with, (list, np.ndarray)):
        transformer_type = type(transform_with[0])
        for transformer in transform_with:
            assert transformer_type == type(transformer), "All transformers in passed list must be of the same type. "

        return transformer_type

    return type(transform_with)


def is_transformer_already_initialized(transform_with: TransformerType | str | type | None) -> bool:
    """Checks whether passed transform_with is already inialized. """

    if transform_with is None:
        return False

    if isinstance(transform_with, (str, TransformerType)):
        return False

    if isinstance(transform_with, type):
        return False

    if isinstance(transform_with, (list, np.ndarray)):
        return True

    return True


def get_transformer_factory(transform_with: TransformerType | str | type | object | None, create_transformer_per_time_series: bool, partial_fit_initialized_transformers: bool) -> TransformerFactory:
    """Creates transformer factory for used transformer. """

    transformer_type = get_transformer_type_or_enum_and_validate(transform_with)
    already_initalized = is_transformer_already_initialized(transform_with)

    transformer_factories = [NoTransformerFactory(), StandardScalerFactory(), L2NormalizerFactory(), LogTransformerFactory(), MaxAbsScalerFactory(),
                             MinMaxScalerFactory(), PowerTransformerFactory(), QuantileTransformerFactory(), RobustScalerFactory(), CustomTransformerFactory(transformer_type)]

    for factory in transformer_factories:
        if factory.can_be_used(transformer_type):
            factory.raise_when_not_supported(create_transformer_per_time_series)

            if already_initalized:
                factory.set_already_initialized_transformers(transform_with)
                factory.raise_when_initialized_not_supported(create_transformer_per_time_series, partial_fit_initialized_transformers)

            return factory

    raise TypeError("Passed transformer type cannot be used! Either use built-in transformer or pass a custom transformer that subclasses from Transformer base class.")
