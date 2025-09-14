from abc import ABC, abstractmethod
import inspect

import numpy as np

from cesnet_tszoo.utils.enums import FillerType


class Filler(ABC):
    """
    Base class for data fillers.

    This class serves as the foundation for creating custom fillers. To implement a custom filler, this class must be subclassed and extended.
    Fillers are used to handle missing data in a dataset.

    Example:

        import numpy as np

        class ForwardFiller(Filler):

            def __init__(self, features):
                super().__init__(features)

                self.last_values = None

            def fill(self, batch_values: np.ndarray, existing_indices: np.ndarray, missing_indices: np.ndarray, **kwargs) -> None:
                if len(missing_indices) > 0 and missing_indices[0] == 0 and self.last_values is not None:
                    batch_values[0] = self.last_values
                    missing_indices = missing_indices[1:]

                mask = np.zeros_like(batch_values, dtype=bool)
                mask[missing_indices] = True
                mask = mask.T

                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)

                batch_values = batch_values.T
                batch_values[mask] = batch_values[np.nonzero(mask)[0], idx[mask]]
                batch_values = batch_values.T

                self.last_values = np.copy(batch_values[-1])
    """

    IDENTIFIER = None

    def __init__(self, features):
        super().__init__()

        self.features = features

    @abstractmethod
    def fill(self, batch_values: np.ndarray, existing_indices: np.ndarray, missing_indices: np.ndarray, **kwargs) -> None:
        """Fills missing data in the `batch_values`.

        This method is responsible for filling missing data within a single time series.

        Parameters:
            batch_values: Data of a single time series with shape `(times, features)` excluding IDs.
            existing_indices: Indices in `batch_values` where data is not missing.
            missing_indices: Indices in `batch_values` where data is missing.
            kwargs: first_next_existing_values, first_next_existing_values_distance, default_values 
        """
        ...


class MeanFiller(Filler):
    """
    Fills values from total mean of all previous values.

    Corresponds to enum [`FillerType.MEAN_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `mean_filler`.
    """

    IDENTIFIER = FillerType.MEAN_FILLER.value

    def __init__(self, features):
        super().__init__(features)

        self.averages = np.zeros(len(features), dtype=np.float64)
        self.total_existing_values = 0

    def fill(self, batch_values: np.ndarray, existing_indices: np.ndarray, missing_indices: np.ndarray, **kwargs) -> None:
        self.total_existing_values += len(existing_indices)

        if len(existing_indices) == 0:
            if self.total_existing_values > 0:
                batch_values[:, :][missing_indices] = self.averages
            return

        existing_values_until_now = self.total_existing_values - len(existing_indices)

        total_divisors = np.arange(1 + existing_values_until_now, len(batch_values) + 1 + existing_values_until_now)

        missing_mask = np.zeros_like(total_divisors)
        missing_mask[missing_indices] = 1

        total_divisors -= np.cumsum(missing_mask, axis=0)

        if total_divisors[0] == 0:
            missing_start_mask = np.logical_and(missing_mask, total_divisors <= 0)
            missing_start_offset = len(batch_values[missing_start_mask])
            total_divisors = total_divisors[missing_start_offset:]
            missing_indices = np.logical_and(missing_mask[missing_start_offset:], total_divisors > 0)
            batch_values[missing_start_offset:][missing_indices] = 0
            new_sums = np.cumsum(batch_values[missing_start_offset:], axis=0)
        else:
            batch_values[missing_indices] = 0
            missing_start_offset = 0
            new_sums = np.cumsum(batch_values[:], axis=0)

        for i in range(len(batch_values[0])):

            updated_old_averages = existing_values_until_now * (self.averages[i] / total_divisors)

            new_averages = new_sums[:, i] / total_divisors + updated_old_averages
            batch_values[missing_start_offset:, i][missing_indices] = new_averages[missing_indices]

            self.averages[i] = new_averages[-1]


class ForwardFiller(Filler):
    """
    Fills missing values based on last existing value. 

    Corresponds to enum [`FillerType.FORWARD_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `forward_filler`.
    """

    IDENTIFIER = FillerType.FORWARD_FILLER.value

    def __init__(self, features):
        super().__init__(features)

        self.last_values = None

    def fill(self, batch_values: np.ndarray, existing_indices: np.ndarray, missing_indices: np.ndarray, **kwargs) -> None:
        if len(missing_indices) > 0 and missing_indices[0] == 0 and self.last_values is not None:
            batch_values[0] = self.last_values
            missing_indices = missing_indices[1:]

        mask = np.zeros_like(batch_values, dtype=bool)
        mask[missing_indices] = True
        mask = mask.T

        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)

        batch_values = batch_values.T
        batch_values[mask] = batch_values[np.nonzero(mask)[0], idx[mask]]
        batch_values = batch_values.T

        self.last_values = np.copy(batch_values[-1])


class LinearInterpolationFiller(Filler):
    """
    Fills values with linear interpolation. 

    Corresponds to enum [`FillerType.LINEAR_INTERPOLATION_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `linear_interpolation_filler`.
    """

    IDENTIFIER = FillerType.LINEAR_INTERPOLATION_FILLER.value

    def __init__(self, features):
        super().__init__(features)

        self.last_values = None
        self.last_values_x_pos = None

    def fill(self, batch_values: np.ndarray, existing_indices: np.ndarray, missing_indices: np.ndarray, **kwargs) -> None:

        if missing_indices is None:
            self.last_values = np.copy(batch_values[-1, :])
            return
        elif len(existing_indices) == 0 and (self.last_values is None or kwargs['first_next_existing_values'] is None):
            return

        if self.last_values is not None and kwargs['first_next_existing_values'] is not None:
            if len(existing_indices) == 0:
                existing_values = np.vstack((self.last_values, kwargs['first_next_existing_values']))
                existing_values_x = np.hstack((self.last_values_x_pos, kwargs['first_next_existing_values_distance']))
            else:
                existing_values = np.vstack((self.last_values, batch_values[existing_indices, :], kwargs['first_next_existing_values']))
                existing_values_x = np.hstack((self.last_values_x_pos, existing_indices, kwargs['first_next_existing_values_distance']))
        elif self.last_values is not None:
            if len(existing_indices) == 0:
                existing_values = np.reshape(self.last_values, (1, len(batch_values[0])))
                existing_values_x = [self.last_values_x_pos]
            else:
                existing_values = np.vstack((self.last_values, batch_values[existing_indices, :]))
                existing_values_x = np.hstack((self.last_values_x_pos, existing_indices))
        elif kwargs['first_next_existing_values'] is not None:
            existing_values = np.vstack((batch_values[existing_indices, :], kwargs['first_next_existing_values']))
            existing_values_x = np.hstack((existing_indices, kwargs['first_next_existing_values_distance']))
        else:
            existing_values = batch_values[existing_indices].view()
            existing_values_x = existing_indices

        for i in range(len(batch_values[0])):
            if len(existing_indices) == 0:
                batch_values[:, i][missing_indices] = np.interp(missing_indices, existing_values_x, existing_values[:, i], left=kwargs["default_values"][i], right=kwargs["default_values"][i])
            else:
                batch_values[:, i][missing_indices] = np.interp(missing_indices, existing_values_x, existing_values[:, i], left=kwargs["default_values"][i], right=kwargs["default_values"][i])

        self.last_values = np.copy(batch_values[-1, :])
        self.last_values_x_pos = -1


class NoFiller(Filler):
    """
    Does nothing. 

    Corresponds to enum [`FillerType.NO_FILLER`][cesnet_tszoo.utils.enums.FillerType] or `None`.
    """

    IDENTIFIER = FillerType.NO_FILLER.value

    def fill(self, batch_values: np.ndarray, existing_indices: np.ndarray, missing_indices: np.ndarray, **kwargs) -> None:
        ...


class FillerFactory(ABC):
    """Base class for filler factories. """

    def __init__(self, filler_type: type, creates_built_in: bool = True):
        self.filler_type = filler_type
        self.creates_built_in = creates_built_in

    @abstractmethod
    def create_filler(self, features) -> Filler:
        """Creates filler instance. """
        ...

    def can_be_used(self, filler_from_input: FillerType | type | None) -> bool:
        """Checks whether factory can be used for passed filler. """

        if isinstance(filler_from_input, FillerType):
            return filler_from_input.value == self.filler_type.IDENTIFIER

        return self.filler_type == filler_from_input


class MeanFillerFactory(FillerFactory):
    """Factory class for MeanFiller. """

    def __init__(self):
        super().__init__(MeanFiller)

    def create_filler(self, features) -> MeanFiller:
        return MeanFiller(features)


class ForwardFillerFactory(FillerFactory):
    """Factory class for ForwardFiller. """

    def __init__(self):
        super().__init__(ForwardFiller)

    def create_filler(self, features) -> ForwardFiller:
        return ForwardFiller(features)


class LinearInterpolationFillerFactory(FillerFactory):
    """Factory class for LinearInterpolationFiller. """

    def __init__(self):
        super().__init__(LinearInterpolationFiller)

    def create_filler(self, features) -> LinearInterpolationFiller:
        return LinearInterpolationFiller(features)


class NoFillerFactory(FillerFactory):
    """Factory class for NoFiller. """

    def __init__(self):
        super().__init__(NoFiller)

    def create_filler(self, features) -> NoFiller:
        return NoFiller(features)

    def can_be_used(self, filler_from_input: FillerType | type | None) -> bool:
        return filler_from_input is None or super().can_be_used(filler_from_input)


class CustomFillerFactory(FillerFactory):
    """Factory class for custom fillers. """

    def __init__(self, filler_type: type):
        super().__init__(filler_type, creates_built_in=False)

        if self.can_be_used(filler_type) and filler_type.IDENTIFIER is None:
            filler_type.IDENTIFIER = f"{self.filler_type.__name__} (Custom)"

    def create_filler(self, features) -> Filler:
        custom_filler = self.filler_type(features)

        return custom_filler

    def can_be_used(self, filler_from_input: type):
        return filler_from_input is not None and inspect.isclass(filler_from_input) and issubclass(filler_from_input, Filler)


def get_filler_factory(fill_missing_with: FillerType | str | type | None) -> FillerFactory:
    """Creates filler factory for used filler. """

    # Validate and process missing data filler type
    if isinstance(fill_missing_with, (str, FillerType)):
        fill_missing_with = FillerType(fill_missing_with)

    filler_factories = [NoFillerFactory(), MeanFillerFactory(), ForwardFillerFactory(), LinearInterpolationFillerFactory(), CustomFillerFactory(fill_missing_with)]
    for factory in filler_factories:
        if factory.can_be_used(fill_missing_with):
            return factory

    raise TypeError("Passed filler type cannot be used! Either use built-in fillers or pass a custom filler that subclasses from Filler base class.")
