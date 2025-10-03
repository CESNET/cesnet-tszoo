from typing import Literal, overload, Union

from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.cesnet_database import CesnetDatabase
import cesnet_tszoo.datasets.databases.cesnet_timeseries24.constants as timeseries24_constants
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class CESNET_TimeSeries24(CesnetDatabase):
    """
    Dataset class for [CESNET_TimeSeries24][cesnet-timeseries24]. 

    Use class method [`get_dataset`][cesnet_tszoo.datasets.CESNET_TimeSeries24.get_dataset] to create a dataset instance.
    """
    name = "CESNET-TimeSeries24"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=cesnet-timeseries24"
    id_names = timeseries24_constants.ID_NAMES
    default_values = timeseries24_constants.DEFAULT_VALUES
    source_types = timeseries24_constants.SOURCE_TYPES
    aggregations = timeseries24_constants.AGGREGATIONS
    additional_data = timeseries24_constants.ADDITIONAL_DATA

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["ip_addresses_sample", "ip_addresses_full", "institution_subnets", "institutions"],
                    aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.TIME_BASED, "time_based"], check_errors: bool = False, display_details: bool = False) -> TimeBasedCesnetDataset: ...

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["ip_addresses_sample", "ip_addresses_full", "institution_subnets", "institutions"],
                    aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.SERIES_BASED, "series_based"], check_errors: bool = False, display_details: bool = False) -> SeriesBasedCesnetDataset: ...

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["ip_addresses_sample", "ip_addresses_full", "institution_subnets", "institutions"],
                    aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.DISJOINT_TIME_BASED, "disjoint_time_based"], check_errors: bool = False, display_details: bool = False) -> DisjointTimeBasedCesnetDataset: ...

    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["ip_addresses_sample", "ip_addresses_full", "institution_subnets", "institutions"], aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: DatasetType | Literal["time_based", "series_based", "disjoint_time_based"], check_errors: bool = False, display_details: bool = False) -> Union[TimeBasedCesnetDataset, SeriesBasedCesnetDataset, DisjointTimeBasedCesnetDataset]:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            source_type: The source type of the desired dataset.
            aggregation: The aggregation type for the selected source type.
            dataset_type: Type of a dataset you want to create. Can be [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset], [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] or [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset].
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`

        Returns:
            [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset], [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] or [`DisjointTimeBasedCesnetDataset`][cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset].
        """

        return super(CESNET_TimeSeries24, cls).get_dataset(data_root, source_type, aggregation, dataset_type, check_errors, display_details)
