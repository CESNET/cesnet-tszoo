from abc import ABC, abstractmethod


class Holder(ABC):

    @abstractmethod
    def get_instance(self, idx: int) -> object:
        ...

    @abstractmethod
    def create_split_copy(self, split_range: slice) -> "Holder":
        """Creates copy with splitted values. """
        ...
