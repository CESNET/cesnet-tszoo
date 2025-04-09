import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from cesnet_tszoo.utils.enums import TimeFormat
from cesnet_tszoo.utils.constants import ID_TIME_COLUMN_NAME, DATETIME_TIME_FORMAT


def collate_fn_simple(batch):
    return batch


def load_from_dataloader(dataloader: DataLoader, ids: np.ndarray[np.uint32], time_seperated: bool, silent: bool = False, is_series_based: bool = False) -> list[np.ndarray]:
    """ Returns all data from dataloader. """

    if not silent:
        print("Loading data from dataloader")

    if is_series_based:
        data = [None for _ in ids]
        times = None

        i = 0

        for batch in tqdm(dataloader, disable=silent):
            batch_times = None
            if time_seperated:
                times = batch[1]
                batch = batch[0]
            for _, element in enumerate(batch):
                data[i] = element
                i += 1
    else:
        data = [[] for _ in ids]
        times = []

        for batch in tqdm(dataloader, disable=silent):
            batch_times = None
            if time_seperated:
                batch_times = batch[1]
                batch = batch[0]
            for i, element in enumerate(batch):
                data[i].append(element)

            times.append(batch_times)
        for i, _ in enumerate(data):
            if len(data[i]) > 0:
                data[i] = np.concatenate(data[i])

        if time_seperated:
            times = np.concatenate(times)

    return {"data": data, "time": times}


def create_single_df_from_dataloader(dataloader: DataLoader, ids: np.ndarray[np.uint32], feature_names: list[str], time_format: TimeFormat, include_element_id: bool, include_time: bool, silent: bool = False, is_series_based: bool = False) -> pd.DataFrame:
    """ Returns data from dataloader as one Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). """

    loaded_data = load_from_dataloader(dataloader, ids, time_seperated=time_format == TimeFormat.DATETIME and include_time, silent=silent, is_series_based=is_series_based)
    merged_data = np.concatenate(loaded_data['data'])
    df = pd.DataFrame(merged_data[:], columns=feature_names)
    if time_format == TimeFormat.DATETIME and include_time:
        df[DATETIME_TIME_FORMAT] = np.tile(loaded_data['time'], len(ids))

        cols = df.columns.tolist()

        if include_element_id:
            cols = cols[:1] + cols[-1:] + cols[1:-1]
        else:
            cols = cols[-1:] + cols[:-1]

        df = df[cols]
    elif include_time:
        df.rename(columns={ID_TIME_COLUMN_NAME: time_format.value}, inplace=True)

    return df


def create_multiple_df_from_dataloader(dataloader: DataLoader, ids: np.ndarray[np.uint32], feature_names: list[str], time_format: TimeFormat, include_element_id: bool, include_time: bool, silent: bool = False, is_per_series: bool = False) -> list[pd.DataFrame]:
    """ Returns data from dataloader as one Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) per time series."""

    loaded_data = load_from_dataloader(dataloader, ids, time_seperated=time_format == TimeFormat.DATETIME and include_time, silent=silent, is_series_based=is_per_series)
    dataframes = []
    for _, element in enumerate(loaded_data["data"]):
        df = pd.DataFrame(element[:], columns=feature_names)
        if time_format == TimeFormat.DATETIME and include_time:
            df[time_format.value] = loaded_data["time"]

            cols = df.columns.tolist()

            if include_element_id:
                cols = cols[:1] + cols[-1:] + cols[1:-1]
            else:
                cols = cols[-1:] + cols[:-1]

            df = df[cols]
        elif include_time:
            df.rename(columns={ID_TIME_COLUMN_NAME: time_format.value}, inplace=True)

        dataframes.append(df)

    return dataframes


def create_numpy_from_dataloader(dataloader: DataLoader, ids: np.ndarray[np.uint32], time_format: TimeFormat, include_time: bool, silent: bool = False, is_per_series: bool = False) -> np.ndarray:
    """ Returns data from dataloader as `np.ndarray`."""

    loaded_data = load_from_dataloader(dataloader, ids, time_seperated=time_format == TimeFormat.DATETIME and include_time, silent=silent, is_series_based=is_per_series)
    arrays = np.ndarray((len(loaded_data["data"]), len(loaded_data["data"][0]), len(loaded_data["data"][0][0])), dtype=loaded_data["data"][0].dtype)

    for i, element in enumerate(loaded_data["data"]):
        arrays[i] = element

    if time_format == TimeFormat.DATETIME and include_time:
        return arrays, loaded_data["time"]
    else:
        return arrays
