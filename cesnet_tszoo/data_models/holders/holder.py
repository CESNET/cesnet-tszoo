from abc import ABC, abstractmethod

import numpy as np


class Holder(ABC):

    @abstractmethod
    def get_instance(self, idx: int) -> object:
        ...

    @abstractmethod
    def update_instance(self, update_with: object, idx: int):
        ...

    @abstractmethod
    def supported_ts_updated(self, supported_ts_ids: np.ndarray):
        ...

    @abstractmethod
    def is_empty(self) -> bool:
        ...

    @abstractmethod
    def fit(self, data: np.ndarray, idx: int):
        ...

    @abstractmethod
    def apply(self, data: np.ndarray, idx: int, **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    def create_split_copy(self, split_range: slice) -> "Holder":
        """Creates copy with splitted values. """
        ...
