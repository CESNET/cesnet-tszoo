from typing import Literal, overload, Union

from cesnet_tszoo.datasets.databases.database_factory import DatabaseFactory
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.SDN import SDN
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class SDNFactory(DatabaseFactory):
    """Dataset factory for SDN database. """

    def __init__(self):
        super().__init__(SDN.name)

    def create_dataset(self, data_root: str, subset: str, aggregation: AgreggationType, dataset_type: DatasetType,
                       check_errors: bool, display_details: bool) -> Union[TimeBasedCesnetDataset, SeriesBasedCesnetDataset, DisjointTimeBasedCesnetDataset]:

        return SDN.get_dataset(data_root, subset, SourceType.MININET_SIMULATOR, aggregation, dataset_type, check_errors, display_details)
