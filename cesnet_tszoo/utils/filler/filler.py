from abc import ABC, abstractmethod
from cesnet_tszoo.utils.constants import BASE_DATA_DTYPE_PART

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

                return batch_values
            """

    @abstractmethod
    def fill(self, batch_values: np.ndarray, missing_masks: dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Fills missing data in the `batch_values`.

        This method is responsible for filling missing data within a single time series.

        Parameters:
            batch_values: Data of a single time series with shape `(times, features)` excluding IDs.
            missing_mask: Mask of missing values in batch_values.
            kwargs: first_next_existing_values, first_next_existing_values_distance, default_values 
        """
        ...


class MeanFiller(Filler):
    """
    Fills values from total mean of all previous values.

    Corresponds to enum [`FillerType.MEAN_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `mean_filler`.
    """

    def __init__(self):
        self.initialized = False
        self.averages = {}
        self.total_existing_values = {}

    def __init_attributes(self, batch_values: np.ndarray):
        for name in batch_values.dtype.names:
            base_shape = batch_values[name].shape[1:]
            self.averages[name] = np.zeros((1, *base_shape), dtype=np.float64)
            self.total_existing_values[name] = np.zeros((1, *base_shape), dtype=np.float64)

        self.initialized = True

    def __fill_section(self, values: np.ndarray, missing_mask: np.ndarray, averages: np.ndarray, total_existing_values: np.ndarray, name: str) -> np.ndarray:
        existing_mask = ~missing_mask
        batch_counts = np.cumsum(existing_mask, axis=0)
        batch_sums = np.cumsum(np.where(existing_mask, values, 0.0), axis=0)

        prev_counts = total_existing_values
        total_counts = batch_counts + prev_counts

        with np.errstate(invalid="ignore", divide="ignore"):
            running_avg = ((averages / total_counts) * prev_counts) + (batch_sums / total_counts)

        running_avg = np.vstack([averages, running_avg[:-1]])

        fill_mask = missing_mask & (total_counts > 0)
        values[fill_mask] = running_avg[fill_mask]

        total_existing_values = total_counts[-1]

        valid_cols = total_existing_values > 0

        self.averages[name][0, valid_cols] = running_avg[-1, valid_cols]
        self.total_existing_values[name] = total_existing_values

        return values

    def fill(self, batch_values: np.ndarray, missing_masks: dict[str, np.ndarray], **kwargs) -> np.ndarray:

        if not self.initialized:
            self.__init_attributes(batch_values)

        for name in batch_values.dtype.names:

            values = batch_values[name].view()
            missing_mask = missing_masks[name]
            averages = self.averages[name]
            total_existing_values = self.total_existing_values[name]

            self.__fill_section(values, missing_mask, averages, total_existing_values, name)

        return batch_values


class ForwardFiller(Filler):
    """
    Fills missing values based on last existing value. 

    Corresponds to enum [`FillerType.FORWARD_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `forward_filler`.
    """

    def __init__(self):
        self.last_values = {}
        self.initialized = False

    def __init_attributes(self, batch_values: np.ndarray):
        for name in batch_values.dtype.names:
            self.last_values[name] = None

        self.initialized = True

    def __fill_section(self, values: np.ndarray, missing_mask: np.ndarray, last_values: np.ndarray, name: str) -> np.ndarray:
        if last_values is not None and np.any(missing_mask[0]):
            values[0, missing_mask[0]] = last_values[missing_mask[0]]

        orig_shape = values.shape
        t = orig_shape[0]
        flat_size = int(np.prod(orig_shape[1:]))

        values_2d = values.reshape(t, flat_size)
        mask_2d = missing_mask.reshape(t, flat_size)

        mask = mask_2d.T
        values_t = values_2d.T

        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)

        values_t[mask] = values_t[np.nonzero(mask)[0], idx[mask]]
        values_t = values_t.T

        values = values_2d.reshape(orig_shape)

        self.last_values[name] = np.copy(values[-1])

        return values

    def fill(self, batch_values: np.ndarray, missing_masks: dict[str, np.ndarray], **kwargs) -> np.ndarray:

        if not self.initialized:
            self.__init_attributes(batch_values)

        for name in batch_values.dtype.names:

            values = batch_values[name].view()
            missing_mask = missing_masks[name]
            last_values = self.last_values[name]

            self.__fill_section(values, missing_mask, last_values, name)

        return batch_values


# TO-DO support for matrices
class LinearInterpolationFiller(Filler):
    """
    Fills values with linear interpolation. 

    Corresponds to enum [`FillerType.LINEAR_INTERPOLATION_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `linear_interpolation_filler`.
    """

    def __init__(self):
        self.last_values = {}
        self.last_values_x_pos = {}
        self.initialized = False

    def __init_attributes(self, batch_values: np.ndarray):
        for name in batch_values.dtype.names:
            self.last_values[name] = None
            self.last_values_x_pos[name] = None

        self.initialized = True

    def __flatten(self, values: np.ndarray, mask: np.ndarray, last_values: np.ndarray | None) -> tuple[np.ndarray]:
        orig_shape = values.shape
        t = orig_shape[0]
        flat_size = int(np.prod(orig_shape[1:]))

        values_2d = values.reshape(t, flat_size)
        mask_2d = mask.reshape(t, flat_size)

        if last_values is not None:
            last_values = last_values.reshape(flat_size)

        return values_2d, mask_2d, orig_shape, last_values

    def __restore_shape(self, values_2d: np.ndarray, orig_shape: tuple) -> np.ndarray:
        return values_2d.reshape(orig_shape)

    def __fill_section(self, values: np.ndarray, missing_mask: np.ndarray, last_values: np.ndarray | None, last_values_x_pos: np.ndarray, default_values: np.ndarray, name: str) -> np.ndarray:
        values, missing_mask, orig_shape, last_values = self.__flatten(values, missing_mask, last_values)
        t, flat_size = values.shape
        x_positions = np.arange(t)

        existing_mask = ~missing_mask
        any_existing = np.any(existing_mask, axis=0)
        any_missing = np.any(missing_mask, axis=0)

        if not np.any(any_missing):
            self.last_values[name] = np.copy(values[-1, :])
            self.last_values_x_pos[name] = -1
            return

        if not np.any(any_existing) and last_values is None:
            return

        for i in range(flat_size):
            existing_x = x_positions[existing_mask[:, i]]
            existing_y = values[existing_mask[:, i], i]

            if last_values is not None:
                existing_x = np.insert(existing_x, 0, last_values_x_pos)
                existing_y = np.insert(existing_y, 0, last_values[i])

            missing_x = x_positions[missing_mask[:, i]]

            if len(missing_x) > 0 and len(existing_x) > 0:
                values[missing_mask[:, i], i] = np.interp(
                    missing_x,
                    existing_x,
                    existing_y,
                    left=default_values[i],
                    right=default_values[i],
                )

        values = self.__restore_shape(values, orig_shape)

        self.last_values[name] = np.copy(values[-1, :])
        self.last_values_x_pos[name] = -1

        return values

    def fill(self, batch_values: np.ndarray, missing_masks: dict[str, np.ndarray], **kwargs) -> np.ndarray:

        if not self.initialized:
            self.__init_attributes(batch_values)

        for name in batch_values.dtype.names:

            default_values = kwargs["default_values"][name]
            values = batch_values[name].view()
            missing_mask = missing_masks[name]
            last_values = self.last_values[name]
            last_values_x_pos = self.last_values_x_pos[name]

            batch_values[name] = self.__fill_section(values, missing_mask, last_values, last_values_x_pos, default_values, name)

        return batch_values


class NoFiller(Filler):
    """
    Does nothing. 

    Corresponds to enum [`FillerType.NO_FILLER`][cesnet_tszoo.utils.enums.FillerType] or literal `no_filler`.
    """

    def fill(self, batch_values: np.ndarray, missing_masks: dict[str, np.ndarray], **kwargs) -> np.ndarray:
        return batch_values
