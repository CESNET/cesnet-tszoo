import numpy as np
import tables as tb


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


def get_additional_data(dataset_path: str, data_name: str) -> np.ndarray:
    """Return additional data dataset. """
    with tb.open_file(dataset_path, mode="r") as dataset:
        additional_data = dataset.get_node(f"/{data_name}")[:]
        return additional_data
