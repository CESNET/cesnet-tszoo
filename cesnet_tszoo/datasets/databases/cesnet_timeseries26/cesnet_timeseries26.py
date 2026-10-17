from typing import Literal, overload, Union

from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.cesnet_database import CesnetDatabase
import cesnet_tszoo.datasets.databases.cesnet_timeseries26.constants as timeseries26_constants
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType
from cesnet_tszoo.utils.enums import DownloadSource


class CESNET_TimeSeries26(CesnetDatabase):
    """
    Dataset class for [CESNET_TimeSeries26][cesnet_timeseries26_page]. 

    Use class method [`get_dataset`][cesnet_tszoo.datasets.CESNET_TimeSeries26.get_dataset] to create a dataset instance.

    Ten source types (institutions, institution subnets, IP addresses: full / random sample / five samples stratified
    by the share of missing values / one subnet monitored since 2021) with 10-minute, 1-hour and 1-day aggregations.
    Files are downloaded from the CESNET DataZoo bucket shared with CESNET-TimeSeries24 first; when the bucket does not
    serve a file (the files are not uploaded there yet) the download falls back to the primary Google Drive, then the
    backup Google Drive, then Zenodo (`fallback_download_urls`). Every file is verified against `checksums`.
    """
    name = "CESNET-TimeSeries26"
    bucket_url = timeseries26_constants.BUCKET_URL
    download_urls = {}
    fallback_download_urls = timeseries26_constants.FALLBACK_DOWNLOAD_URLS
    checksums = timeseries26_constants.CHECKSUMS
    id_names = timeseries26_constants.ID_NAMES
    default_values = timeseries26_constants.DEFAULT_VALUES
    source_types = timeseries26_constants.SOURCE_TYPES
    aggregations = timeseries26_constants.AGGREGATIONS

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["institutions", "institution_subnets", "ip_addresses_full", "ip_addresses_sample", "ip_addresses_sample_gaps_20_00", "ip_addresses_sample_gaps_40_20", "ip_addresses_sample_gaps_60_40", "ip_addresses_sample_gaps_80_60", "ip_addresses_sample_gaps_100_80", "subnet_longitudinal"],
                    aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.TIME_BASED, "time_based"], check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> TimeBasedCesnetDataset: ...

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["institutions", "institution_subnets", "ip_addresses_full", "ip_addresses_sample", "ip_addresses_sample_gaps_20_00", "ip_addresses_sample_gaps_40_20", "ip_addresses_sample_gaps_60_40", "ip_addresses_sample_gaps_80_60", "ip_addresses_sample_gaps_100_80", "subnet_longitudinal"],
                    aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.SERIES_BASED, "series_based"], check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> SeriesBasedCesnetDataset: ...

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["institutions", "institution_subnets", "ip_addresses_full", "ip_addresses_sample", "ip_addresses_sample_gaps_20_00", "ip_addresses_sample_gaps_40_20", "ip_addresses_sample_gaps_60_40", "ip_addresses_sample_gaps_80_60", "ip_addresses_sample_gaps_100_80", "subnet_longitudinal"],
                    aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.DISJOINT_TIME_BASED, "disjoint_time_based"], check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> DisjointTimeBasedCesnetDataset: ...

    @classmethod
    def get_dataset(cls, data_root: str, source_type: SourceType | Literal["institutions", "institution_subnets", "ip_addresses_full", "ip_addresses_sample", "ip_addresses_sample_gaps_20_00", "ip_addresses_sample_gaps_40_20", "ip_addresses_sample_gaps_60_40", "ip_addresses_sample_gaps_80_60", "ip_addresses_sample_gaps_100_80", "subnet_longitudinal"], aggregation: AgreggationType | Literal["10_minutes", "1_hour", "1_day"],
                    dataset_type: DatasetType | Literal["time_based", "series_based", "disjoint_time_based"], check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> Union[TimeBasedCesnetDataset, SeriesBasedCesnetDataset, DisjointTimeBasedCesnetDataset]:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            source_type: The source type of the desired dataset.
            aggregation: The aggregation type for the selected source type.
            dataset_type: Type of a dataset you want to create. Can be [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) or [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset).
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`
            download_source: Where to download the dataset from when it is not in `data_root` yet: `"auto"` (every configured source in order), `"datazoo"`, `"google_drive"` or `"zenodo"`. See [`DownloadSource`][cesnet_tszoo.utils.enums.DownloadSource]. `Default: "auto"`

        Returns:
            [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) or [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset).
        """

        return super(CESNET_TimeSeries26, cls).get_dataset(data_root, None, source_type, aggregation, dataset_type, check_errors, display_details, download_source=download_source)
