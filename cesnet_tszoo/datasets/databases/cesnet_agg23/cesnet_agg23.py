from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.cesnet_database import CesnetDatabase
import cesnet_tszoo.datasets.databases.cesnet_agg23.constants as agg23_constants
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType
from cesnet_tszoo.utils.enums import DownloadSource


class CESNET_AGG23(CesnetDatabase):
    """
    Dataset class for [CESNET_AGG23][cesnet-agg23]. 

    Files are downloaded from CESNET DataZoo, with the Google Drive copies in `fallback_download_urls` as a secondary source, and verified against `checksums`.

    Use class method [`get_dataset`][cesnet_tszoo.datasets.CESNET_AGG23.get_dataset] to create a dataset instance.
    """
    name = "CESNET-AGG23"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=cesnet-agg23"
    fallback_download_urls = agg23_constants.FALLBACK_DOWNLOAD_URLS
    checksums = agg23_constants.CHECKSUMS
    id_names = agg23_constants.ID_NAMES
    default_values = agg23_constants.DEFAULT_VALUES
    source_types = agg23_constants.SOURCE_TYPES
    aggregations = agg23_constants.AGGREGATIONS

    @classmethod
    def get_dataset(cls, data_root: str, check_errors: bool = False, display_details: bool = False, download_source: DownloadSource | str = "auto") -> TimeBasedCesnetDataset:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`
            download_source: Where to download the dataset from when it is not in `data_root` yet: `"auto"` (every configured source in order), `"datazoo"`, `"google_drive"` or `"zenodo"`. See [`DownloadSource`][cesnet_tszoo.utils.enums.DownloadSource]. `Default: "auto"`

        Returns:
            TimeBasedCesnetDataset
        """

        return super(CESNET_AGG23, cls).get_dataset(data_root, None, SourceType.CESNET2, AgreggationType.AGG_1_MINUTE, DatasetType.TIME_BASED, check_errors, display_details, backward_compatibility=True, download_source=download_source)
