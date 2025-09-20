from abc import ABC, abstractmethod
import inspect

import numpy as np

from cesnet_tszoo.utils.enums import TransformerType
from cesnet_tszoo.utils.transformer import Transformer, MinMaxScaler, StandardScaler, L2Normalizer, LogTransformer, MaxAbsScaler, PowerTransformer, QuantileTransformer, RobustScaler, NoTransformer
from cesnet_tszoo.utils.transformer import transformer_has_fit_method, transformer_has_partial_fit_method, transformer_has_transform, get_transformer_type_or_enum_and_validate, is_transformer_already_initialized


class TransformerFactory(ABC):
    """Base class for transformer factories. """

    def __init__(self, transformer_type: type, identifier: TransformerType | None, can_fit: bool, can_partial_fit: bool, creates_built_in: bool = True, is_empty_factory: bool = False):
        self.transformer_type = transformer_type
        self.can_fit = can_fit
        self.can_partial_fit = can_partial_fit
        self.creates_built_in = creates_built_in
        self.has_already_initialized = False
        self.initialized_transformers = None
        self.is_empty_factory = is_empty_factory
        self.has_single_initialized = None
        self.identifier = identifier

        if isinstance(transformer_type, type):
            self.name = transformer_type.__name__

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

    def can_be_used(self, transform_with: TransformerType | type) -> bool:
        """Checks whether factory can be used for passed transformer. """

        if isinstance(transform_with, TransformerType):
            return transform_with == self.identifier

        return isinstance(transform_with, type) and self.transformer_type == transform_with

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

        assert transformer_has_transform(self.transformer_type), "Transformer must have method transform"

        if self.has_already_initialized:
            return

        if create_transformer_per_time_series:
            assert self.can_fit, "Fit must be supported if you want to create transformer per time series."

        if not create_transformer_per_time_series:
            assert self.can_partial_fit, "Partial fit must be supported if you want to use one transformer for multiple time series."


class StandardScalerFactory(TransformerFactory):
    """Factory class for StandardScaler. """

    def __init__(self):
        super().__init__(StandardScaler, TransformerType.STANDARD_SCALER, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> StandardScaler:
        return StandardScaler()


class L2NormalizerFactory(TransformerFactory):
    """Factory class for L2Normalizer. """

    def __init__(self):
        super().__init__(L2Normalizer, TransformerType.L2_NORMALIZER, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> L2Normalizer:
        return L2Normalizer()


class LogTransformerFactory(TransformerFactory):
    """Factory class for LogTransformer. """

    def __init__(self):
        super().__init__(LogTransformer, TransformerType.LOG_TRANSFORMER, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> LogTransformer:
        return LogTransformer()


class MaxAbsScalerFactory(TransformerFactory):
    """Factory class for MaxAbsScaler. """

    def __init__(self):
        super().__init__(MaxAbsScaler, TransformerType.MAX_ABS_SCALER, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> MaxAbsScaler:
        return MaxAbsScaler()


class MinMaxScalerFactory(TransformerFactory):
    """Factory class for MinMaxScaler. """

    def __init__(self):
        super().__init__(MinMaxScaler, TransformerType.MIN_MAX_SCALER, can_fit=True, can_partial_fit=True)

    def create_transformer(self) -> MinMaxScaler:
        return MinMaxScaler()


class PowerTransformerFactory(TransformerFactory):
    """Factory class for PowerTransformer. """

    def __init__(self):
        super().__init__(PowerTransformer, TransformerType.POWER_TRANSFORMER, can_fit=True, can_partial_fit=False)

    def create_transformer(self) -> PowerTransformer:
        return PowerTransformer()


class QuantileTransformerFactory(TransformerFactory):
    """Factory class for QuantileTransformer. """

    def __init__(self):
        super().__init__(QuantileTransformer, TransformerType.QUANTILE_TRANSFORMER, can_fit=True, can_partial_fit=False)

    def create_transformer(self) -> QuantileTransformer:
        return QuantileTransformer()


class RobustScalerFactory(TransformerFactory):
    """Factory class for RobustScaler. """

    def __init__(self):
        super().__init__(RobustScaler, TransformerType.ROBUST_SCALER, can_fit=True, can_partial_fit=False)

    def create_transformer(self) -> RobustScaler:
        return RobustScaler()


class NoTransformerFactory(TransformerFactory):
    """Factory class for NoTransformer. """

    def __init__(self):
        super().__init__(NoTransformer, TransformerType.NO_TRANSFORMER, can_fit=True, can_partial_fit=True, is_empty_factory=True)

    def create_transformer(self) -> NoTransformer:
        return NoTransformer()


class CustomTransformerFactory(TransformerFactory):
    """Factory class for custom transformer. """

    def __init__(self, transform_with: type):
        super().__init__(transform_with, None, can_fit=None, can_partial_fit=None, creates_built_in=False)

        if self.can_be_used(transform_with):
            self.name = f"{transform_with.__name__} (Custom)"
            self.can_fit = transformer_has_fit_method(transform_with)
            self.can_partial_fit = transformer_has_partial_fit_method(transform_with)

    def create_transformer(self) -> Transformer:
        return self.transformer_type()

    def can_be_used(self, transform_with: type) -> bool:
        return isinstance(transform_with, type) and inspect.isclass(transform_with)


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
