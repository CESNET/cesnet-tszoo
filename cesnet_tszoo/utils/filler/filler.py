from abc import ABC, abstractmethod

import numpy as np


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

    def __init__(self, features):
        super().__init__()

        self.features = features

    @abstractmethod
    def fill(self, batch_values: np.ndarray, missing_mask: np.ndarray, **kwargs) -> None:
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

    def __init__(self, features):
        super().__init__(features)

        self.averages = np.zeros(len(features), dtype=np.float64)
        self.total_existing_values = np.zeros(len(features), dtype=np.float64)

    def fill(self, batch_values: np.ndarray, missing_mask: np.ndarray, **kwargs) -> None:

        existing_mask = ~missing_mask
        batch_counts = np.cumsum(existing_mask, axis=0)
        batch_sums = np.cumsum(np.where(existing_mask, batch_values, 0.0), axis=0)

        prev_counts = self.total_existing_values
        total_counts = batch_counts + prev_counts

        with np.errstate(invalid="ignore", divide="ignore"):
            running_avg = ((self.averages / total_counts) * prev_counts) + (batch_sums / total_counts)

        running_avg = np.vstack([self.averages, running_avg[:-1, :]])

        fill_mask = missing_mask & (total_counts > 0)
        batch_values[fill_mask] = running_avg[fill_mask]

        self.total_existing_values = total_counts[-1]
        valid_cols = self.total_existing_values > 0
        self.averages[valid_cols] = running_avg[-1, valid_cols]


class ForwardFiller(Filler):
    """
    Fills missing values based on last existing value. 

    Corresponds to enum [`FillerType.FORWARD_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `forward_filler`.
    """

    def __init__(self, features):
        super().__init__(features)

        self.last_values = None

    def fill(self, batch_values: np.ndarray, missing_mask: np.ndarray, **kwargs) -> None:
        if self.last_values is not None and np.any(missing_mask[0]):
            batch_values[0, missing_mask[0]] = self.last_values[missing_mask[0]]

        mask = missing_mask.T

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

    def __init__(self, features):
        super().__init__(features)

        self.last_values = None
        self.last_values_x_pos = None

    def fill(self, batch_values: np.ndarray, missing_mask: np.ndarray, **kwargs) -> None:

        default_values = kwargs["default_values"]

        existing_mask = ~missing_mask
        any_existing = np.any(existing_mask, axis=0)
        any_missing = np.any(missing_mask, axis=0)

        no_missing = not np.any(any_missing)
        no_existing = not np.any(any_existing)

        if no_missing:
            self.last_values = np.copy(batch_values[-1, :])
            return

        if no_existing and self.last_values is None:
            return

        x_positions = np.arange(batch_values.shape[0])

        for i in range(batch_values.shape[1]):
            existing_x = x_positions[existing_mask[:, i]]
            existing_y = batch_values[existing_mask[:, i], i]

            if self.last_values is not None:
                existing_x = np.insert(existing_x, 0, self.last_values_x_pos)
                existing_y = np.insert(existing_y, 0, self.last_values[i])

            missing_x = x_positions[missing_mask[:, i]]

            if len(missing_x) > 0 and len(existing_x) > 0:
                batch_values[missing_mask[:, i], i] = np.interp(
                    missing_x,
                    existing_x,
                    existing_y,
                    left=default_values[i],
                    right=default_values[i],
                )

        self.last_values = np.copy(batch_values[-1, :])
        self.last_values_x_pos = -1


class NoFiller(Filler):
    """
    Does nothing. 

    Corresponds to enum [`FillerType.NO_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `no_filler`.
    """

    def fill(self, batch_values: np.ndarray, missing_mask: np.ndarray, **kwargs) -> None:
        ...
