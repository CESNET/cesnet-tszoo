from abc import ABC, abstractmethod


class Holder(ABC):

    @abstractmethod
    def get_instance(self, idx: int) -> object:
        ...
