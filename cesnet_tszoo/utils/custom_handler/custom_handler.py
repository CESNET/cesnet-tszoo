from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from cesnet_tszoo.utils.enums import SplitType


class CustomHandler(ABC):

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def get_target_sets() -> set[SplitType] | set[Literal["train", "val", "test", "all"]]:
        ...


class PerSeriesCustomHandler(CustomHandler):

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        ...

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def get_target_sets() -> set[SplitType] | set[Literal["train", "val", "test", "all"]]:
        ...


class AllSeriesCustomHandler(CustomHandler):

    @abstractmethod
    def partial_fit(self, data: np.ndarray) -> None:
        ...

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def get_target_sets() -> set[SplitType] | set[Literal["train", "val", "test", "all"]]:
        ...


class NoFitCustomHandler(CustomHandler):

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def get_target_sets() -> set[SplitType] | set[Literal["train", "val", "test", "all"]]:
        ...
