import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from cesnet_tszoo.utils.enums import TimeFormat, DatasetType
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, DATETIME_TIME_FORMAT, TIME_DTYPE_PART, BASE_DATA_DTYPE_PART


def collate_fn_simple(batch):
    return batch


def create_single_df_from_dataloader(dataloader: DataLoader, ids: np.ndarray[np.uint32], feature_names: list[str], time_format: TimeFormat, include_element_id: bool, include_time: bool, dataset_type: DatasetType, silent: bool = False) -> pd.DataFrame:
    """ Returns data from dataloader as one Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). """

    loaded_data = __load_from_dataloader(dataloader, ids, silent=silent, dataset_type=dataset_type)
    merged_data = np.concatenate(loaded_data)
    df = __create_dataframe(merged_data, len(ids), feature_names, time_format, include_element_id, include_time)

    return df


def create_multiple_df_from_dataloader(dataloader: DataLoader, ids: np.ndarray[np.uint32], feature_names: list[str], time_format: TimeFormat, include_element_id: bool, include_time: bool, dataset_type: DatasetType, silent: bool = False) -> list[pd.DataFrame]:
    """ Returns data from dataloader as one Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) per time series."""

    loaded_data = __load_from_dataloader(dataloader, ids, silent=silent, dataset_type=dataset_type)
    dataframes = []
    for one_ts_data in loaded_data:
        df = __create_dataframe(one_ts_data, 1, feature_names, time_format, include_element_id, include_time)

        dataframes.append(df)

    return dataframes


def create_numpy_from_dataloader(dataloader: DataLoader, ids: np.ndarray[np.uint32], dataset_type: DatasetType, silent: bool = False) -> np.ndarray:
    """ Returns data from dataloader as `np.ndarray`."""

    loaded_data = __load_from_dataloader(dataloader, ids, silent=silent, dataset_type=dataset_type)

    return np.stack(loaded_data, axis=0)


def __load_from_dataloader(dataloader: DataLoader, ids: np.ndarray[np.uint32], dataset_type: DatasetType, silent: bool = False) -> list[np.ndarray]:
    """ Returns all data from dataloader. """

    if not silent:
        print("Loading data from dataloader")

    if dataset_type == DatasetType.SERIES_BASED:
        data = [None for _ in ids]
        times = None

        i = 0

        for batch in tqdm(dataloader, disable=silent):
            batch_times = None
            for _, element in enumerate(batch):
                data[i] = element
                i += 1
    else:
        data = [[] for _ in ids]
        times = []

        for batch in tqdm(dataloader, disable=silent):
            batch_times = None
            for i, element in enumerate(batch):
                data[i].append(element)

            times.append(batch_times)
        for i, _ in enumerate(data):
            if len(data[i]) > 0:
                data[i] = np.concatenate(data[i])

    return data


def __create_dataframe(data: np.ndarray, ts_count: int, feature_names: list[str], time_format: TimeFormat, include_element_id: bool, include_time: bool) -> pd.DataFrame:
    df = pd.DataFrame([], columns=feature_names)

    for name in data.dtype.names:
        if name == TIME_DTYPE_PART:
            continue

        if name == BASE_DATA_DTYPE_PART:
            df[[column for column in df.columns if column not in data.dtype.names]] = data[BASE_DATA_DTYPE_PART]
        else:
            df[name] = data[name].tolist()

    if time_format == TimeFormat.DATETIME and include_time:
        df[DATETIME_TIME_FORMAT] = np.tile(data[TIME_DTYPE_PART], ts_count)

        cols = df.columns.tolist()

        if include_element_id:
            cols = cols[:1] + cols[-1:] + cols[1:-1]
        else:
            cols = cols[-1:] + cols[:-1]

        df = df[cols]
    elif include_time:
        df.rename(columns={ID_TIME_COLUMN_NAME: time_format.value}, inplace=True)

    return df
