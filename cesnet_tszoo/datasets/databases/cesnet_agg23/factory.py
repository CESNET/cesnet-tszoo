from cesnet_tszoo.datasets.databases.database_factory import DatabaseFactory
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.cesnet_agg23 import CESNET_AGG23
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class CESNET_AGG23Factory(DatabaseFactory):
    """Dataset factory for CESNET_AGG23 database. """

    def __init__(self):
        super().__init__(CESNET_AGG23.name)

    def create_dataset(self, data_root: str, source_type: SourceType | str, aggregation: AgreggationType | str, dataset_type: DatasetType | str, check_errors: bool, display_details: bool) -> TimeBasedCesnetDataset:
        return CESNET_AGG23.get_dataset(data_root, check_errors, display_details)
