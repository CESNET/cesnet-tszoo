from dataclasses import dataclass, field
from copy import copy, deepcopy

import numpy as np

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.custom_handler.custom_handler import CustomHandler, PerSeriesCustomHandler, AllSeriesCustomHandler, NoFitCustomHandler


@dataclass
class PerSeriesCustomHandlerHolder(Holder):
    custom_handlers: np.ndarray[PerSeriesCustomHandler] = field(init=True)

    def get_instance(self, idx: int) -> PerSeriesCustomHandler:

        if self.custom_handlers is None:
            raise ValueError()

        if idx > len(self.custom_handlers):
            raise ValueError()

        return self.custom_handlers[idx]

    def is_empty(self):
        return self.custom_handlers is None

    def create_split_copy(self, split_range: slice) -> "PerSeriesCustomHandlerHolder":

        split_copy = copy(self)

        split_copy.custom_handlers = np.array(self.custom_handlers[split_range])

        return split_copy

    def update_instance(self, update_with: PerSeriesCustomHandler, idx: int):
        self.custom_handlers[idx] = update_with

    def fit(self, data: np.ndarray, idx: int):
        self.get_instance(idx).fit(data)

    def apply(self, data: np.ndarray, idx: int, **kwargs) -> np.ndarray:
        return self.get_instance(idx).apply(data)

    def supported_ts_updated(self, supported_ts_ids: np.ndarray):
        if self.is_empty():
            return

        self.custom_handlers = self.custom_handlers[supported_ts_ids]


@dataclass
class AllSeriesCustomHandlerHolder(Holder):
    custom_handler: AllSeriesCustomHandler = field(init=True)

    def get_instance(self, idx: int) -> AllSeriesCustomHandler:

        if self.custom_handler is None:
            raise ValueError()

        return self.custom_handler

    def is_empty(self):
        return self.custom_handler is None

    def create_split_copy(self, split_range: slice) -> "AllSeriesCustomHandlerHolder":

        split_copy = copy(self)

        split_copy.custom_handler = self.custom_handler

        return split_copy

    def update_instance(self, update_with: AllSeriesCustomHandler, idx: int):
        self.custom_handler = update_with

    def fit(self, data: np.ndarray, idx: int):
        self.get_instance(idx).partial_fit(data)

    def supported_ts_updated(self, supported_ts_ids: np.ndarray):
        ...

    def apply(self, data: np.ndarray, idx: int, **kwargs) -> np.ndarray:
        return self.get_instance(idx).apply(data)


@dataclass
class NoFitCustomHandlerHolder(Holder):
    custom_handlers: np.ndarray[NoFitCustomHandler] = field(init=True)

    def get_instance(self, idx: int) -> NoFitCustomHandler:

        if self.custom_handlers is None:
            raise ValueError()

        if idx > len(self.custom_handlers):
            raise ValueError()

        return self.custom_handlers[idx]

    def is_empty(self):
        return self.custom_handlers is None

    def create_split_copy(self, split_range: slice) -> "NoFitCustomHandlerHolder":

        split_copy = copy(self)

        split_copy.custom_handlers = np.array(self.custom_handlers[split_range])

        return split_copy

    def update_instance(self, update_with: NoFitCustomHandler, idx: int):
        self.custom_handlers[idx] = update_with

    def fit(self, data: np.ndarray, idx: int):
        ...

    def supported_ts_updated(self, supported_ts_ids: np.ndarray):
        if self.is_empty():
            return

        self.custom_handlers = self.custom_handlers[supported_ts_ids]

    def apply(self, data: np.ndarray, idx: int, **kwargs) -> np.ndarray:
        return self.get_instance(idx).apply(data)
