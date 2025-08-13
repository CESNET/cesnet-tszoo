from dataclasses import dataclass
from typing import Optional

from cesnet_tszoo.utils.enums import AgreggationType, SourceType, DatasetType
import cesnet_tszoo.version as version
from packaging.version import Version


def get_abbreviated_list_string(to_abbreviate, max_length: int = 5):
    """Returns visual shortend version of possibly big list. """

    if to_abbreviate is None:
        return None

    if len(to_abbreviate) <= max_length * 2:
        return f"{to_abbreviate}, Length={len(to_abbreviate)}"
    else:
        beggining = str(to_abbreviate[:max_length]).removesuffix(']')
        ending = str(to_abbreviate[-max_length:]).removeprefix('[')
        return f"{beggining} ... {ending}, Length={len(to_abbreviate)}"


def is_config_used_for_dataset(config, dataset_database: str, dataset_source_type: SourceType, dataset_aggregation: AgreggationType) -> bool:
    """Checks whether config can be used for dataset. """
    if config.database_name != dataset_database:
        return False

    if config.source_type != dataset_source_type:
        return False

    if config.aggregation != dataset_aggregation:
        return False

    return True


@dataclass
class ExportBenchmark:
    """Used for exporting benchmark. """

    database_name: str
    source_type: SourceType
    aggregation: AgreggationType
    dataset_type: str
    config_identifier: str
    annotations_ts_identifier: str
    annotations_time_identifier: str
    annotations_both_identifier: str
    related_results_identifier: Optional[str] = None
    version: str = None
    is_series_based: Optional[bool] = None
    description: Optional[str] = None

    def to_dict(self):
        """Converts class to dict. For export support. """

        deprecated = ["is_series_based"]

        return {key: value for key, value in self.__dict__.items() if key not in deprecated}

    @classmethod
    def from_dict(cls, data):
        """Converts to class from dict. For import support. """

        data = cls._update_for_backward_compatibility(data)

        return cls(**data)

    @classmethod
    def _update_for_backward_compatibility(cls, data):

        if "version" not in data:
            data["version"] = version.DEFAULT_VERSION

        if Version(data["version"]) < Version(version.VERSION_0_1_3):
            del data["is_series_based"]
            data["dataset_type"] = None

        data["version"] = version.current_version

        return data
