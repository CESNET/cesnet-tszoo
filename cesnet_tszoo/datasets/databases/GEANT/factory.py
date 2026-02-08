from typing import Literal, overload, Union

from cesnet_tszoo.datasets.databases.database_factory import DatabaseFactory
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.GEANT import GEANT
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class GEANTFactory(DatabaseFactory):
    """Dataset factory for GEANT database. """

    def __init__(self):
        super().__init__(GEANT.name)

    @overload
    def create_dataset(self, data_root: str, subset: str, dataset_type: Literal[DatasetType.TIME_BASED, "time_based"],
                       check_errors: bool, display_details: bool) -> TimeBasedCesnetDataset: ...

    @overload
    def create_dataset(self, data_root: str, subset: str, dataset_type: Literal[DatasetType.SERIES_BASED, "series_based"],
                       check_errors: bool, display_details: bool) -> SeriesBasedCesnetDataset: ...

    @overload
    def create_dataset(self, data_root: str, subset: str, dataset_type: Literal[DatasetType.DISJOINT_TIME_BASED, "disjoint_time_based"],
                       check_errors: bool, display_details: bool) -> DisjointTimeBasedCesnetDataset: ...

    def create_dataset(self, data_root: str, subset: str, dataset_type: DatasetType | Literal["time_based", "series_based", "disjoint_time_based"],
                       check_errors: bool, display_details: bool) -> Union[TimeBasedCesnetDataset, SeriesBasedCesnetDataset, DisjointTimeBasedCesnetDataset]:

        return GEANT.get_dataset(data_root, subset, SourceType.BACKBONE_NETWORK, AgreggationType.AGG_15_MINUTES, dataset_type, check_errors, display_details)
