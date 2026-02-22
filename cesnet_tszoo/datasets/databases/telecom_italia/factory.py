from typing import Literal, overload, Union

from cesnet_tszoo.datasets.databases.database_factory import DatabaseFactory
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.telecom_italia import TelecomItalia
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class TelecomItaliaFactory(DatabaseFactory):
    """Dataset factory for Telecom Italia database. """

    def __init__(self):
        super().__init__(TelecomItalia.name)

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

        return TelecomItalia.get_dataset(data_root, subset, SourceType.SERVICE_PROVIDER, AgreggationType.AGG_10_MINUTES, dataset_type, check_errors, display_details)
