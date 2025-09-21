from abc import ABC, abstractmethod

from cesnet_tszoo.datasets.cesnet_dataset import CesnetDataset
from cesnet_tszoo.utils.enums import DatasetType, SourceType, AgreggationType
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset


class DatasetFactory(ABC):
    """Base class for dataset factories. """

    def __init__(self, dataset_type: DatasetType):
        self.dataset_type = dataset_type

    @abstractmethod
    def create_dataset(self, database_name: str, dataset_path: str, configs_root: str, benchmarks_root: str, annotations_root: str,
                       source_type: SourceType, aggregation: AgreggationType, ts_id_name: str, default_values: dict, additional_data: dict[str, tuple]) -> CesnetDataset:
        """Creates dataset based on dataset type. """
        ...

    def can_be_used(self, dataset_type_to_create: DatasetType) -> bool:
        """Checks whether factory can be used for passed dataset type. """
        return dataset_type_to_create == self.dataset_type


# Implemented factories

class TimeBasedDatasetFactory(DatasetFactory):
    """Factory class for TimeBasedCesnetDataset. """

    def __init__(self):
        super().__init__(DatasetType.TIME_BASED)

    def create_dataset(self, database_name: str, dataset_path: str, configs_root: str, benchmarks_root: str, annotations_root: str,
                       source_type: SourceType, aggregation: AgreggationType, ts_id_name: str, default_values: dict, additional_data: dict[str, tuple]) -> TimeBasedCesnetDataset:

        return TimeBasedCesnetDataset(database_name, dataset_path, configs_root, benchmarks_root, annotations_root, source_type, aggregation, ts_id_name, default_values, additional_data)


class DisjointTimeBasedDatasetFactory(DatasetFactory):
    """Factory class for DisjointTimeBasedCesnetDataset. """

    def __init__(self):
        super().__init__(DatasetType.DISJOINT_TIME_BASED)

    def create_dataset(self, database_name: str, dataset_path: str, configs_root: str, benchmarks_root: str, annotations_root: str,
                       source_type: SourceType, aggregation: AgreggationType, ts_id_name: str, default_values: dict, additional_data: dict[str, tuple]) -> DisjointTimeBasedCesnetDataset:

        return DisjointTimeBasedCesnetDataset(database_name, dataset_path, configs_root, benchmarks_root, annotations_root, source_type, aggregation, ts_id_name, default_values, additional_data)


class SeriesBasedDatasetFactory(DatasetFactory):
    """Factory class for SeriesBasedCesnetDataset. """

    def __init__(self):
        super().__init__(DatasetType.SERIES_BASED)

    def create_dataset(self, database_name: str, dataset_path: str, configs_root: str, benchmarks_root: str, annotations_root: str,
                       source_type: SourceType, aggregation: AgreggationType, ts_id_name: str, default_values: dict, additional_data: dict[str, tuple]) -> SeriesBasedCesnetDataset:

        return SeriesBasedCesnetDataset(database_name, dataset_path, configs_root, benchmarks_root, annotations_root, source_type, aggregation, ts_id_name, default_values, additional_data)

# Implemented factories


def get_dataset_factory(dataset_type_to_create: DatasetType | str) -> DatasetFactory:
    """Creates dataset factory for specified parameters. """

    dataset_type_to_create = DatasetType(dataset_type_to_create)

    for factory in DatasetFactory.__subclasses__():
        factory_instance = factory()

        if factory_instance.can_be_used(dataset_type_to_create):
            return factory_instance

    raise TypeError("Could not find matching dataset for passed dataset type.")
