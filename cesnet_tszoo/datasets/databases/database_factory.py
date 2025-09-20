from abc import ABC, abstractmethod

from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class DatabaseFactory(ABC):
    """Base class for database factories for creating their datasets. """

    def __init__(self, database_name: str):
        self.database_name = database_name

    @abstractmethod
    def create_dataset(self, data_root: str, source_type: SourceType | str, aggregation: AgreggationType | str, dataset_type: DatasetType | str, check_errors: bool, display_details: bool) -> CesnetDataset:
        """Creates databases dataset based on constructor parameters. """
        ...

    def can_be_used(self, database_to_create_name: str) -> bool:
        """Checks whether factory can be used for passed database name. """
        return database_to_create_name == self.database_name


def get_database_factory(database_name: str) -> DatabaseFactory:
    """Creates database factory for specified parameters. """

    for factory in DatabaseFactory.__subclasses__():
        factory_instance = factory()

        if factory_instance.can_be_used(database_name):
            return factory_instance

    raise TypeError("Could not find matching database for passed database name.")
