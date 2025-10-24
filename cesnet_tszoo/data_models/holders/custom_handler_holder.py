from dataclasses import dataclass, field
from copy import copy

from cesnet_tszoo.data_models.holders.holder import Holder
from cesnet_tszoo.utils.custom_handler.custom_handler import CustomHandler


@dataclass
class PerSeriesCustomHandlerHolder(Holder):
    custom_handlers: list[CustomHandler] = field(init=True)

    def get_instance(self, idx: int) -> CustomHandler:

        if self.custom_handlers is None:
            raise ValueError()

        if idx > len(self.custom_handlers):
            raise ValueError()

        return self.custom_handlers[idx]

    def is_empty(self):
        return self.custom_handlers is None

    def create_split_copy(self, split_range: slice) -> "PerSeriesCustomHandlerHolder":

        split_copy = copy(self)

        split_copy.custom_handlers = list(self.custom_handlers[split_range])

        return split_copy


@dataclass
class AllSeriesCustomHandlerHolder(Holder):
    custom_handler: CustomHandler = field(init=True)

    def get_instance(self, idx: int) -> CustomHandler:

        if self.custom_handler is None:
            raise ValueError()

        return self.custom_handler

    def is_empty(self):
        return self.custom_handler is None

    def create_split_copy(self, split_range: slice) -> "AllSeriesCustomHandlerHolder":

        split_copy = copy(self)

        split_copy.custom_handler = self.custom_handler

        return split_copy


@dataclass
class NoFitCustomHandlerHolder(Holder):
    custom_handlers: list[CustomHandler] = field(init=True)

    def get_instance(self, idx: int) -> CustomHandler:

        if self.custom_handlers is None:
            raise ValueError()

        if idx > len(self.custom_handlers):
            raise ValueError()

        return self.custom_handlers[idx]

    def is_empty(self):
        return self.custom_handlers is None

    def create_split_copy(self, split_range: slice) -> "NoFitCustomHandlerHolder":

        split_copy = copy(self)

        split_copy.custom_handlers = list(self.custom_handlers[split_range])

        return split_copy
