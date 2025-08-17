from typing import Literal

from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.combined_cesnet_dataset import CombinedCesnetDataset
from cesnet_tszoo.datasets.cesnet_database import CesnetDatabase
from cesnet_tszoo.datasets.datasets_constants import _CESNET_TIME_SERIES24_ID_NAMES, _CESNET_TIME_SERIES24_DEFAULT_VALUES, _CESNET_TIME_SERIES24_AGGREGATIONS, _CESNET_TIME_SERIES24_SOURCE_TYPES, _CESNET_AGG23_ID_NAMES, _CESNET_AGG23_DEFAULT_VALUES, _CESNET_AGG23_AGGREGATIONS, _CESNET_AGG23_SOURCE_TYPES, _CESNET_TIME_SERIES24_ADDITIONAL_DATA
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class CESNET_TimeSeries24(CesnetDatabase):
    """
    Dataset class for [CESNET_TimeSeries24][cesnet-timeseries24]. 

    Use class method [`get_dataset`][cesnet_tszoo.datasets.datasets.CESNET_TimeSeries24.get_dataset] to create a dataset instance.
    """
    name = "CESNET-TimeSeries24"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=cesnet-timeseries24"
    id_names = _CESNET_TIME_SERIES24_ID_NAMES
    default_values = _CESNET_TIME_SERIES24_DEFAULT_VALUES
    source_types = _CESNET_TIME_SERIES24_SOURCE_TYPES
    aggregations = _CESNET_TIME_SERIES24_AGGREGATIONS
    additional_data = _CESNET_TIME_SERIES24_ADDITIONAL_DATA

    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["ip_addresses_sample", "ip_addresses_full", "institution_subnets", "institutions"], aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: DatasetType | Literal["time_based", "series_based", "combined"], check_errors: bool = False, display_details: bool = False) -> TimeBasedCesnetDataset | SeriesBasedCesnetDataset | CombinedCesnetDataset:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            source_type: The source type of the desired dataset.
            aggregation: The aggregation type for the selected source type.
            dataset_type: Type of a dataset you want to create. Can be [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset], [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] or [`CombinedCesnetDataset`][cesnet_tszoo.datasets.combined_cesnet_dataset.CombinedCesnetDataset].
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`

        Returns:
            [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset], [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset] or [`CombinedCesnetDataset`][cesnet_tszoo.datasets.combined_cesnet_dataset.CombinedCesnetDataset].
        """

        return super(CESNET_TimeSeries24, cls).get_dataset(data_root, source_type, aggregation, dataset_type, check_errors, display_details)


class CESNET_AGG23(CesnetDatabase):
    """
    Dataset class for [CESNET_AGG23][cesnet-agg23]. 

    Use class method [`get_dataset`][cesnet_tszoo.datasets.datasets.CESNET_AGG23.get_dataset] to create a dataset instance.
    """
    name = "CESNET-AGG23"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=cesnet-agg23"
    id_names = _CESNET_AGG23_ID_NAMES
    default_values = _CESNET_AGG23_DEFAULT_VALUES
    source_types = _CESNET_AGG23_SOURCE_TYPES
    aggregations = _CESNET_AGG23_AGGREGATIONS

    @classmethod
    def get_dataset(cls, data_root: str, check_errors: bool = False, display_details: bool = False) -> TimeBasedCesnetDataset:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`

        Returns:
            TimeBasedCesnetDataset
        """

        return super(CESNET_AGG23, cls).get_dataset(data_root, SourceType.CESNET2, AgreggationType.AGG_1_MINUTE, DatasetType.TIME_BASED, check_errors, display_details)
