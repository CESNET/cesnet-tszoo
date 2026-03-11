from typing import Literal, overload, Union

from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.datasets.series_based_cesnet_dataset import SeriesBasedCesnetDataset
from cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset import DisjointTimeBasedCesnetDataset
from cesnet_tszoo.datasets.databases.cesnet_database import CesnetDatabase
import cesnet_tszoo.datasets.databases.SDN.constants as sdn_constants
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


class SDN(CesnetDatabase):
    """
    Dataset class for [SDN][sdn]. 

    Use class method [`get_dataset`][cesnet_tszoo.datasets.SDN.get_dataset] to create a dataset instance.
    """
    name = "SDN"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=sdn"
    id_names = sdn_constants.ID_NAMES
    default_values = sdn_constants.DEFAULT_VALUES
    matrix_feature_mappings = sdn_constants.MATRIX_FEATURE_MAPPINGS
    subsets = ["matrix", "node", "node2node"]
    source_types = sdn_constants.SOURCE_TYPES
    aggregations = sdn_constants.AGGREGATIONS

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, subset: Literal["matrix", "node", "node2node"], aggregation: AgreggationType | Literal["1_minute", "10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.TIME_BASED, "time_based"], check_errors: bool = False, display_details: bool = False) -> TimeBasedCesnetDataset: ...

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, subset: Literal["matrix", "node", "node2node"], aggregation: AgreggationType | Literal["1_minute", "10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.SERIES_BASED, "series_based"], check_errors: bool = False, display_details: bool = False) -> SeriesBasedCesnetDataset: ...

    @overload
    @classmethod
    def get_dataset(cls, data_root: str, subset: Literal["matrix", "node", "node2node"], aggregation: AgreggationType | Literal["1_minute", "10_minutes", "1_hour", "1_day"],
                    dataset_type: Literal[DatasetType.DISJOINT_TIME_BASED, "disjoint_time_based"], check_errors: bool = False, display_details: bool = False) -> DisjointTimeBasedCesnetDataset: ...

    @classmethod
    def get_dataset(cls, data_root: str, subset: Literal["matrix", "node", "node2node"], aggregation: AgreggationType | Literal["1_minute", "10_minutes", "1_hour", "1_day"],
                    dataset_type: DatasetType | Literal["time_based", "series_based", "disjoint_time_based"], check_errors: bool = False, display_details: bool = False) -> Union[TimeBasedCesnetDataset, SeriesBasedCesnetDataset, DisjointTimeBasedCesnetDataset]:
        """
        Create new dataset instance.

        Parameters:
            data_root: Path to the folder where the dataset will be stored. Each database has its own subfolder `data_root/tszoo/databases/database_name/`.
            subset: Specific subset of the dataset.
            aggregation: The aggregation type for the selected source type.
            dataset_type: Type of a dataset you want to create. Can be [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) or [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset).
            check_errors: Whether to validate if the dataset is corrupted. `Default: False`
            display_details: Whether to display details about the available data in chosen dataset. `Default: False`

        Returns:
            [`TimeBasedCesnetDataset`](reference_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset), [`SeriesBasedCesnetDataset`](reference_series_based_cesnet_dataset.md#cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset) or [`DisjointTimeBasedCesnetDataset`](reference_disjoint_time_based_cesnet_dataset.md#cesnet_tszoo.datasets.disjoint_time_based_cesnet_dataset.DisjointTimeBasedCesnetDataset).
        """

        return super(SDN, cls).get_dataset(data_root, subset, SourceType.MININET_SIMULATOR, aggregation, dataset_type, check_errors, display_details)
