from typing import Literal, overload, Union

from cesnet_tszoo.datasets.databases.database_factory import DatabaseFactory
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.vigo_network_operator_kpis import Vigo_NetworkOperatorKPIs
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class Vigo_NetworkOperatorKPIsFactory(DatabaseFactory):
    """Dataset factory for Vigo_NetworkOperatorKPIs database. """

    def __init__(self):
        super().__init__(Vigo_NetworkOperatorKPIs.name)

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

        return Vigo_NetworkOperatorKPIs.get_dataset(data_root, subset, SourceType.NETWORK_OPERATOR, AgreggationType.AGG_5_MINUTES, dataset_type, check_errors, display_details)
