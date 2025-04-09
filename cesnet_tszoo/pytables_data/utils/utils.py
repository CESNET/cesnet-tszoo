import numpy as np
import tables as tb

from cesnet_tszoo.utils.enums import SourceType, AgreggationType


def load_database(dataset_path: str, table_data_path: str = None, mode: str = "r") -> tuple[tb.File, tb.Node]:
    """Prepare dataset that is ready for use. """

    dataset = tb.open_file(dataset_path, mode=mode,
                           chunk_cache_size=1024 * 1024 * 1024 * 4 * 1)

    table_data = None

    try:
        if table_data_path is not None:
            table_data = dataset.get_node(table_data_path)
    except tb.NoSuchNodeError as e:
        raise e

    return dataset, table_data


def get_time_indices(dataset_path: str, table_time_indices_path: str) -> np.ndarray:
    """Return time indices used in dataset. """

    with tb.open_file(dataset_path, mode="r") as dataset:
        time_indices = dataset.get_node(table_time_indices_path)[:]
        return time_indices


def get_column_names(dataset_path: str, source_type: SourceType, aggregation: AgreggationType) -> list[str]:
    """Return feature names used in dataset. """

    with tb.open_file(dataset_path, mode="r") as dataset:
        table = dataset.get_node(f"/{source_type.value}/{AgreggationType._to_str_with_agg(aggregation)}")

        result = []

        for key in table.coldescrs:
            result.append(key)

        return result


def get_column_types(dataset_path: str, source_type: SourceType, aggregation: AgreggationType) -> dict[str, np.dtype]:
    """Return feature types used in dataset. """

    with tb.open_file(dataset_path, mode="r") as dataset:
        table = dataset.get_node(f"/{source_type.value}/{AgreggationType._to_str_with_agg(aggregation)}")

        result = {}

        for key in table.coldescrs:
            result[key] = table.coldescrs[key].dtype

        return result


def get_ts_indices(dataset_path: str, table_identifiers_path: str) -> np.ndarray:
    """Return time series indices used in dataset. """

    with tb.open_file(dataset_path, mode="r") as dataset:
        element_indices = dataset.get_node(table_identifiers_path)[:]
        return element_indices


def get_additional_data(dataset_path: str, data_name: str) -> np.ndarray:
    """Return additional data dataset. """
    with tb.open_file(dataset_path, mode="r") as dataset:
        additional_data = dataset.get_node(f"/{data_name}")[:]
        return additional_data


def get_ts_row_ranges(dataset_path: str, table_ts_row_ranges_path: str) -> np.ndarray:
    """Return ts row ranges used in dataset. """

    with tb.open_file(dataset_path, mode="r") as dataset:
        ts_row_ranges = dataset.get_node(table_ts_row_ranges_path)[:]
        return ts_row_ranges


def get_table_identifiers_path(source_type: SourceType) -> str:
    """Return path to table in dataset for time series identifiers. """

    return f"/{source_type.value}/identifiers"


def get_table_time_indices_path(aggregation: AgreggationType) -> str:
    """Return path to table in dataset for time identifiers. """

    return f"/times/times_{aggregation.value}"
