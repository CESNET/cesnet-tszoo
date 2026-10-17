from typing import Literal, overload, Union

from cesnet_tszoo.datasets.databases.database_factory import DatabaseFactory
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.cesnet_timeseries26 import CESNET_TimeSeries26
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class CESNET_TimeSeries26Factory(DatabaseFactory):
    """Dataset factory for CESNET_TimeSeries26 database. """

    def __init__(self):
        super().__init__(CESNET_TimeSeries26.name)

    @overload
    def create_dataset(self, data_root: str, subset: str, source_type: SourceType | str, aggregation: AgreggationType | str, dataset_type: Literal[DatasetType.TIME_BASED, "time_based"],
                       check_errors: bool, display_details: bool) -> TimeBasedCesnetDataset: ...

    @overload
    def create_dataset(self, data_root: str, subset: str, source_type: SourceType | str, aggregation: AgreggationType | str, dataset_type: Literal[DatasetType.SERIES_BASED, "series_based"],
                       check_errors: bool, display_details: bool) -> SeriesBasedCesnetDataset: ...

    @overload
    def create_dataset(self, data_root: str, subset: str, source_type: SourceType | str, aggregation: AgreggationType | str, dataset_type: Literal[DatasetType.DISJOINT_TIME_BASED, "disjoint_time_based"],
                       check_errors: bool, display_details: bool) -> DisjointTimeBasedCesnetDataset: ...

    def create_dataset(self, data_root: str, subset: str, source_type: SourceType | str, aggregation: AgreggationType | str, dataset_type: DatasetType | Literal["time_based", "series_based", "disjoint_time_based"],
                       check_errors: bool, display_details: bool) -> Union[TimeBasedCesnetDataset, SeriesBasedCesnetDataset, DisjointTimeBasedCesnetDataset]:

        return CESNET_TimeSeries26.get_dataset(data_root, source_type, aggregation, dataset_type, check_errors, display_details)
