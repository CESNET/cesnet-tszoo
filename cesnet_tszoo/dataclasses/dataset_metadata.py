from dataclasses import dataclass, field
import logging

import numpy as np
import tables as tb

from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType


@dataclass
class DatasetMetadata:
    """Class that holds various metadata used for dataset.

    Attributes:
        database_name: Name of the database.
        dataset_path: Path to the dataset file.     
        configs_root: Path to the folder where configurations are saved.
        benchmarks_root: Path to the folder where benchmarks are saved.
        annotations_root: Path to the folder where annotations are saved.
        source_type: The source type of the dataset.
        aggregation: The aggregation type for the selected source type.
        ts_id_name: Name of the id used for time series.
        default_values: Default values for each available feature.
        additional_data: Available additional small datasets.   
        time_indices: Available time IDs for the dataset.
        ts_indices: Available time series IDs for the dataset.   
        ts_row_ranges: Available time series id ranges. 
        features: Available features for the dataset.
        data_table_path: Path to the main data table of the dataset.
    """

    database_name: str
    dataset_type: DatasetType
    dataset_path: str
    configs_root: str
    benchmarks_root: str
    annotations_root: str
    source_type: SourceType
    aggregation: AgreggationType
    ts_id_name: str
    default_values: dict
    additional_data: dict[str, tuple]

    time_indices: np.ndarray = field(default=None, init=False)
    ts_indices: np.ndarray = field(default=None, init=False)
    ts_row_ranges: np.ndarray = field(default=None, init=False)
    features: dict[str, np.dtype] = field(default=None, init=False)
    data_table_path: str = field(default=None, init=False)

    def __post_init__(self):
        self.logger = logging.getLogger("dataset_metadata")

        self.__init_time_indices()
        self.__init_ts_indices()
        self.__init_ts_row_ranges()
        self.__init_features()
        self.__update_default_values()
        self.__init_table_data_path()

    def __init_time_indices(self):
        """Sets time indices used in dataset. """

        with tb.open_file(self.dataset_path, mode="r") as dataset:
            self.time_indices = dataset.get_node(f"/times/times_{self.aggregation.value}")[:]

        self.logger.debug("Time indices have been successfully set.")

    def __init_ts_indices(self):
        """Sets time series indices used in dataset. """

        with tb.open_file(self.dataset_path, mode="r") as dataset:
            self.ts_indices = dataset.get_node(f"/{self.source_type.value}/identifiers")[:]

        self.logger.debug("Time series indices have been successfully set.")

    def __init_ts_row_ranges(self):
        """Sets ts row ranges used in dataset. """

        with tb.open_file(self.dataset_path, mode="r") as dataset:
            node_path = f"/{self.source_type.value}/id_ranges_{AgreggationType._to_str_with_agg(self.aggregation)}"
            self.ts_row_ranges = dataset.get_node(node_path)[:]

        self.logger.debug("Time series row ranges have been successfully set.")

    def __init_features(self):
        """Sets features used in dataset. """

        with tb.open_file(self.dataset_path, mode="r") as dataset:
            table = dataset.get_node(f"/{self.source_type.value}/{AgreggationType._to_str_with_agg(self.aggregation)}")

            result = {}

            for key in table.coldescrs:
                result[key] = table.coldescrs[key].dtype

            self.features = result

        self.logger.debug("Features have been successfully set.")

    def __update_default_values(self):
        """Updates to only relevant default values """

        self.default_values = {feature: self.default_values[feature] for feature in self.default_values if feature in self.features.keys()}

        self.logger.debug("Default values for features updated.")

    def __init_table_data_path(self) -> str:
        """Sets the path to the data table corresponding to the `source_type` and `aggregation`."""

        self.data_table_path = f"/{self.source_type.value}/{AgreggationType._to_str_with_agg(self.aggregation)}"

        self.logger.debug("Data table path for dataset have been successfully set.")
