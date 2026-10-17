from typing import Literal, overload, Union

from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.cesnet_database import CesnetDatabase
import cesnet_tszoo.datasets.databases.vigo_network_operator_kpis.constants as network_operator_kpis_constants
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType
from cesnet_tszoo.utils.enums import DownloadSource


class Vigo_NetworkOperatorKPIs(CesnetDatabase):
    """
    Dataset class for [Vigo_NetworkOperatorKPIs][vigo-networkoperatorkpis]. 

    Files are downloaded from CESNET DataZoo, with the Google Drive copies in `fallback_download_urls` as a secondary source, and verified against `checksums`.

    Use class method [`get_dataset`][cesnet_tszoo.datasets.Vigo_NetworkOperatorKPIs.get_dataset] to create a dataset instance.
    """
    name = "Vigo-NetworkOperatorKPIs"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=vigo-networkoperatorkpis"
    fallback_download_urls = network_operator_kpis_constants.FALLBACK_DOWNLOAD_URLS
    checksums = network_operator_kpis_constants.CHECKSUMS
    id_names = network_operator_kpis_constants.ID_NAMES
    default_values = network_operator_kpis_constants.DEFAULT_VALUES
    subsets = ["downstream", "internet", "sessions", "vpn"]
    source_types = network_operator_kpis_constants.SOURCE_TYPES
    aggregations = network_operator_kpis_constants.AGGREGATIONS

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, subset: Literal["downstream", "internet", "sessions", "vpn"], aggregation: AgreggationType | Literal["5_minutes", "10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.TIME_BASED, "time_based"], check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> TimeBasedCesnetDataset: ...

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, subset: Literal["downstream", "internet", "sessions", "vpn"], aggregation: AgreggationType | Literal["5_minutes", "10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.SERIES_BASED, "series_based"], check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> SeriesBasedCesnetDataset: ...

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, subset: Literal["downstream", "internet", "sessions", "vpn"], aggregation: AgreggationType | Literal["5_minutes", "10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.DISJOINT_TIME_BASED, "disjoint_time_based"], check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> DisjointTimeBasedCesnetDataset: ...

    @classmethod
    def get_dataset(cls, data_root: str, subset: Literal["downstream", "internet", "sessions", "vpn"], aggregation: AgreggationType | Literal["5_minutes", "10_minutes", "1_hour", "1_day"],
                    dataset_type: DatasetType | Literal["time_based", "series_based", "disjoint_time_based"], check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> Union[TimeBasedCesnetDataset, SeriesBasedCesnetDataset, DisjointTimeBasedCesnetDataset]:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            subset: Specific subset of the dataset.
            aggregation: The aggregation type for the selected source type.
            dataset_type: Type of a dataset you want to create. Can be [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) or [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset).
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`
            download_source: Where to download the dataset from when it is not in `data_root` yet: `"auto"` (every configured source in order), `"datazoo"`, `"google_drive"` or `"zenodo"`. See [`DownloadSource`][cesnet_tszoo.utils.enums.DownloadSource]. `Default: "auto"`

        Returns:
            [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) or [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset).
        """

        return super(Vigo_NetworkOperatorKPIs, cls).get_dataset(data_root, subset, SourceType.NETWORK_OPERATOR, aggregation, dataset_type, check_errors, display_details, download_source=download_source)
