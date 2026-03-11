from abc import ABC, abstractmethod

import numpy as np
import sklearn.preprocessing as sk

from cesnet_tszoo.utils.enums import TransformerType


class Transformer(ABC):
    """
    Base class for transformers, used for transforming data.

    This class serves as the foundation for creating custom transformers. To implement a custom transformer, this class is recommended to be subclassed and extended.

    Example:

        import numpy as np

        class LogTransformer(Transformer):

            def __init__(self):
                self.names = None

            def fit(self, data: np.ndarray) -> None:
                self.partial_fit(data)

            def partial_fit(self, data: np.ndarray) -> None:
                self.names = data.dtype.names

            def transform(self, data: np.ndarray) -> np.ndarray:

                for name in data.dtype.names:

                    current_data = data[name]
                    log_data = np.ma.log(current_data)
                    current_data[:] = log_data.filled(np.nan)

                return data

            def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:

                names = transformed_data.dtype.names if transformed_data.dtype.names is not None else self.names

                for name in names:

                    current_data = transformed_data[name] if transformed_data.dtype.names is not None else transformed_data
                    current_data[:] = np.exp(current_data)

                return transformed_data               
    """

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Sets the transformer values for a given time series part.

        This method must be implemented if using multiple transformers that have not been pre-fitted.

        Parameters:
            data: A structured numpy array representing data for a single time series with shape `(times)`. Use data["base_data"] to get non matrix features excluding any identifiers. 
                  For matrix features use their name instead of base_data.
        """
        ...

    @abstractmethod
    def partial_fit(self, data: np.ndarray) -> None:
        """
        Partially sets the transformer values for a given time series part.

        This method must be implemented if using a single transformer that is not pre-fitted for all time series, or when using pre-fitted transformer(s) with `partial_fit_initialized_transformers` set to `True`.

        Parameters:
            data: A structured numpy array representing data for a single time series with shape `(times)`. Use data["base_data"] to get non matrix features excluding any identifiers. 
                  For matrix features use their name instead of base_data.   
        """
        ...

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the input data for a given time series part.

        This method must always be implemented.

        Parameters:
            data: A structured numpy array representing data for a single time series with shape `(times)`. Use data["base_data"] to get non matrix features excluding any identifiers. 
                  For matrix features use their name instead of base_data.

        Returns:
            The transformed data, with the same shape and dtype as the input `(times)`.            
        """
        ...

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """
        Transforms the input transformed data to their original representation for a given time series part.

        Parameters:
            transformed_data: A structured numpy array representing data for a single time series with shape `(times)`. Use data["base_data"] to get non matrix features excluding any identifiers. 
                              For matrix features use their name instead of base_data. 

        Returns:
            The original representation of transformed data, with the same shape and dtype as the input `(times)`.            
        """
        return transformed_data


class MinMaxScaler(Transformer):
    """
    Tranforms data using Scikit [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

    Corresponds to enum [`TransformerType.MIN_MAX_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `min_max_scaler`.
    """

    def __init__(self):
        self.transformers: dict[str, sk.MinMaxScaler] = {}

    def fit(self, data: np.ndarray) -> None:
        self.partial_fit(data)

    def partial_fit(self, data: np.ndarray) -> None:

        is_init = len(self.transformers) == 0

        for name in data.dtype.names:

            if is_init:
                transformer = self.transformers[name] = sk.MinMaxScaler()
            else:
                transformer = self.transformers[name]

            current_data = data[name]
            flat_size = int(np.prod(current_data.shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            transformer.partial_fit(current_data)

    def transform(self, data: np.ndarray) -> np.ndarray:

        for name in data.dtype.names:

            transformer = self.transformers[name]
            current_data = data[name]
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            current_data[:] = transformer.transform(current_data)

        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:

        names = transformed_data.dtype.names if transformed_data.dtype.names is not None else self.transformers.keys()

        for name in names:

            transformer = self.transformers[name]
            current_data = transformed_data[name] if transformed_data.dtype.names is not None else transformed_data
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            current_data[:] = transformer.inverse_transform(current_data)

        return transformed_data


class StandardScaler(Transformer):
    """
    Tranforms data using Scikit [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

    Corresponds to enum [`TransformerType.STANDARD_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `standard_scaler`.
    """

    def __init__(self):
        self.transformers: dict[str, sk.StandardScaler] = {}

    def fit(self, data: np.ndarray) -> None:
        self.partial_fit(data)

    def partial_fit(self, data: np.ndarray) -> None:

        is_init = len(self.transformers) == 0

        for name in data.dtype.names:

            if is_init:
                transformer = self.transformers[name] = sk.StandardScaler()
            else:
                transformer = self.transformers[name]

            current_data = data[name]
            flat_size = int(np.prod(current_data.shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            transformer.partial_fit(current_data)

    def transform(self, data: np.ndarray) -> np.ndarray:

        for name in data.dtype.names:

            transformer = self.transformers[name]
            current_data = data[name]
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            current_data[:] = transformer.transform(current_data)

        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:

        names = transformed_data.dtype.names if transformed_data.dtype.names is not None else self.transformers.keys()

        for name in names:

            transformer = self.transformers[name]
            current_data = transformed_data[name] if transformed_data.dtype.names is not None else transformed_data
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            current_data[:] = transformer.inverse_transform(current_data)

        return transformed_data


class MaxAbsScaler(Transformer):
    """
    Tranforms data using Scikit [`MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).

    Corresponds to enum [`TransformerType.MAX_ABS_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `max_abs_scaler`.
    """

    def __init__(self):
        self.transformers: dict[str, sk.MaxAbsScaler] = {}

    def fit(self, data: np.ndarray) -> None:
        self.partial_fit(data)

    def partial_fit(self, data: np.ndarray) -> None:

        is_init = len(self.transformers) == 0

        for name in data.dtype.names:

            if is_init:
                transformer = self.transformers[name] = sk.MaxAbsScaler()
            else:
                transformer = self.transformers[name]

            current_data = data[name]
            flat_size = int(np.prod(current_data.shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            transformer.partial_fit(current_data)

    def transform(self, data: np.ndarray) -> np.ndarray:

        for name in data.dtype.names:

            transformer = self.transformers[name]
            current_data = data[name]
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            current_data[:] = transformer.transform(current_data)

        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:

        names = transformed_data.dtype.names if transformed_data.dtype.names is not None else self.transformers.keys()

        for name in names:

            transformer = self.transformers[name]
            current_data = transformed_data[name] if transformed_data.dtype.names is not None else transformed_data
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)

            current_data[:] = transformer.inverse_transform(current_data)

        return transformed_data


class LogTransformer(Transformer):
    """
    Tranforms data with natural logarithm. Zero or invalid values are set to `np.nan`.

    Corresponds to enum [`TransformerType.LOG_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `log_transformer`.
    """

    def __init__(self):
        self.names = None

    def fit(self, data: np.ndarray) -> None:
        self.partial_fit(data)

    def partial_fit(self, data: np.ndarray) -> None:
        self.names = data.dtype.names

    def transform(self, data: np.ndarray) -> np.ndarray:

        for name in data.dtype.names:

            current_data = data[name]
            log_data = np.ma.log(current_data)
            current_data[:] = log_data.filled(np.nan)

        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:

        names = transformed_data.dtype.names if transformed_data.dtype.names is not None else self.names

        for name in names:

            current_data = transformed_data[name] if transformed_data.dtype.names is not None else transformed_data
            current_data[:] = np.exp(current_data)

        return transformed_data


class L2Normalizer(Transformer):
    """
    Tranforms data using Scikit [`L2Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html).

    Corresponds to enum [`TransformerType.L2_NORMALIZER`][cesnet_tszoo.utils.enums.TransformerType] or literal `l2_normalizer`.
    """

    def __init__(self):
        self.transformer = sk.Normalizer(norm="l2")

    def fit(self, data: np.ndarray) -> None:
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        for name in data.dtype.names:
            current_data = data[name]
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            current_data[:] = self.transformer.transform(current_data)

        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Normalizer does not support inverse_transform.")


class RobustScaler(Transformer):
    """
    Tranforms data using Scikit [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

    Corresponds to enum [`TransformerType.ROBUST_SCALER`][cesnet_tszoo.utils.enums.TransformerType] or literal `robust_scaler`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.    
    """

    def __init__(self):
        self.transformers: dict[str, sk.RobustScaler] = {}

    def fit(self, data: np.ndarray) -> None:

        for name in data.dtype.names:
            transformer = self.transformers[name] = sk.RobustScaler()
            current_data = data[name]
            flat_size = int(np.prod(current_data.shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            transformer.fit(current_data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("RobustScaler does not support partial_fit.")

    def transform(self, data: np.ndarray) -> np.ndarray:

        for name in data.dtype.names:
            transformer = self.transformers[name]
            current_data = data[name]
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            current_data[:] = transformer.transform(current_data)

        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:

        names = transformed_data.dtype.names if transformed_data.dtype.names is not None else self.transformers.keys()

        for name in names:
            transformer = self.transformers[name]
            current_data = transformed_data[name] if transformed_data.dtype.names is not None else transformed_data
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            current_data[:] = transformer.inverse_transform(current_data)

        return transformed_data


class PowerTransformer(Transformer):
    """
    Tranforms data using Scikit [`PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html).

    Corresponds to enum [`TransformerType.POWER_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `power_transformer`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.
    """

    def __init__(self):
        self.transformers: dict[str, sk.PowerTransformer] = {}

    def fit(self, data: np.ndarray) -> None:

        for name in data.dtype.names:
            transformer = self.transformers[name] = sk.PowerTransformer()
            current_data = data[name]
            flat_size = int(np.prod(current_data.shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            transformer.fit(current_data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("PowerTransformer does not support partial_fit.")

    def transform(self, data: np.ndarray) -> np.ndarray:

        for name in data.dtype.names:
            transformer = self.transformers[name]
            current_data = data[name]
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            current_data[:] = transformer.transform(current_data)

        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:

        names = transformed_data.dtype.names if transformed_data.dtype.names is not None else self.transformers.keys()

        for name in names:
            transformer = self.transformers[name]
            current_data = transformed_data[name] if transformed_data.dtype.names is not None else transformed_data
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            current_data[:] = transformer.inverse_transform(current_data)

        return transformed_data


class QuantileTransformer(Transformer):
    """
    Tranforms data using Scikit [`QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html).

    Corresponds to enum [`TransformerType.QUANTILE_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `quantile_transformer`.

    !!! warning "partial_fit not supported"
        Because this transformer does not support partial_fit it can't be used when using one transformer that needs to be fitted for multiple time series.    
    """

    def __init__(self):
        self.transformers: dict[str, sk.QuantileTransformer] = {}

    def fit(self, data: np.ndarray) -> None:

        for name in data.dtype.names:
            transformer = self.transformers[name] = sk.QuantileTransformer()
            current_data = data[name]
            flat_size = int(np.prod(current_data.shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            transformer.fit(current_data)

    def partial_fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("QuantileTransformer does not support partial_fit.")

    def transform(self, data: np.ndarray) -> np.ndarray:

        for name in data.dtype.names:
            transformer = self.transformers[name]
            current_data = data[name]
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            current_data[:] = transformer.transform(current_data)

        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:

        names = transformed_data.dtype.names if transformed_data.dtype.names is not None else self.transformers.keys()

        for name in names:
            transformer = self.transformers[name]
            current_data = transformed_data[name] if transformed_data.dtype.names is not None else transformed_data
            original_shape = current_data.shape
            flat_size = int(np.prod(original_shape[1:]))

            current_data = current_data.reshape(current_data.shape[0], flat_size)
            current_data[:] = transformer.inverse_transform(current_data)

        return transformed_data


class NoTransformer(Transformer):
    """
    Does nothing.

    Corresponds to enum [`TransformerType.NO_TRANSFORMER`][cesnet_tszoo.utils.enums.TransformerType] or literal `no_transformer`.
    """

    def fit(self, data: np.ndarray):
        ...

    def partial_fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        return transformed_data


def transformer_has_fit_method(to_check) -> bool:
    """Checks whether `to_check` has fit method. """

    fit_method = getattr(to_check, "fit", None)
    if callable(fit_method):
        return True

    return False


def transformer_has_partial_fit_method(to_check) -> bool:
    """Checks whether `to_check` has partial_fit method. """

    partial_fit_method = getattr(to_check, "partial_fit", None)
    if callable(partial_fit_method):
        return True

    return False


def transformer_has_transform(to_check) -> bool:
    """Checks wheter type has transform method. """

    transform_method = getattr(to_check, "transform", None)
    if callable(transform_method):
        return True

    return False


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
