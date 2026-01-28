import pandas as pd
import numpy as np
import os
import tqdm
import tables
from pathlib import Path
import gc
import yaml
import warnings
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Dataset and output configuration")

parser.add_argument(
    "dataset_path",
    type=str,
    help="Path to the dataset directory"
)

parser.add_argument(
    "output_h5_path",
    type=str,
    help="Path where the H5 files will be saved"
)

parser.add_argument(
    "output_h5_base_name",
    type=str,
    help="Base name for the output H5 files"
)

args = parser.parse_args()

DATASET_PATH = args.dataset_path
OUTPUT_H5_PATH = args.output_h5_path
OUTPUT_H5_BASE_NAME = args.output_h5_base_name

IDENTIFIERS_NAME = "identifiers"
IDENTIFIERS = f"{IDENTIFIERS_NAME}.csv"
TS_ID = "ts_id"

TIMES_NAME = "times"
TIMES = f"{TIMES_NAME}.csv"

TS_INFO = "ts_info.yaml"

MAIN_DATA = "main_data"
ADDITIONAL_DATA = "additional_data"

types_mapping_to_np = {

    "float": np.float64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,

    "int": np.int64,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,

    "uint": np.uint64,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,

    "bool": np.bool_,

    "time": np.int64,
}

types_mapping_to_pytables = {
    "float": tables.Float64Col,
    "float16": tables.Float16Col,
    "float32": tables.Float32Col,
    "float64": tables.Float64Col,

    "int": tables.Int64Col,
    "int16": tables.Int16Col,
    "int32": tables.Int32Col,
    "int64": tables.Int64Col,

    "uint": tables.UInt64Col,
    "uint16": tables.UInt16Col,
    "uint32": tables.UInt32Col,
    "uint64": tables.UInt64Col,

    "bool": tables.BoolCol,

    "time": tables.Time32Col,
}


def get_np_type(to_parse: str) -> np.dtype:
    to_parse = to_parse.strip().lower()
    if to_parse in types_mapping_to_np:
        return types_mapping_to_np[to_parse]
    elif "str" == to_parse[:3]:
        return np.string_
    else:
        raise NotImplementedError("Invalid type")


def get_pytables_init_type(to_parse: str, pos: int) -> tables.DTypeLike:
    to_parse = to_parse.strip().replace(" ", "").lower()
    if to_parse in types_mapping_to_np:
        tb_type = types_mapping_to_pytables[to_parse]
        if tb_type == tables.Time32Col:
            return tb_type(pos=pos, dflt=-1)
        else:
            return tb_type(pos=pos)

    elif "str" == to_parse[:3]:
        splits = to_parse.split(",")

        if len(splits) != 2:
            raise NotImplementedError("Invalid type")

        return tables.StringCol(int(splits[1]), pos=pos)
    else:
        raise NotImplementedError("Invalid type")


class IdRanges(tables.IsDescription):
    ts_id = tables.UInt32Col(pos=0)
    start = tables.UInt64Col(pos=1)
    end = tables.UInt64Col(pos=2)


class Identifiers(tables.IsDescription):
    ts_id = tables.UInt32Col(pos=0)


class TimesDataset(tables.IsDescription):
    id_time = tables.UInt32Col(pos=0)
    time = tables.Time32Col(pos=1, dflt=-1)


class Outputer:
    def __init__(self, ts_type: str, source_type: str, aggregation_type: str, input_path: str, output_path: str):
        self.base_name = OUTPUT_H5_BASE_NAME.lower()
        self.ts_type = ts_type.lower()
        self.source_type = source_type.lower()
        self.aggregation_type = aggregation_type.lower()
        self.name = f"{self.base_name}-{self.ts_type}-{self.source_type}-{self.aggregation_type}"
        self.file_name = f"{self.name}.h5"

        self.main_data_group_name = source_type
        self.main_data_table_name = f"agg_{aggregation_type}"
        self.id_ranges_table_name = f"id_ranges_agg_{aggregation_type}"
        self.times_group_name = TIMES_NAME
        self.times_table_name = f"{TIMES_NAME}_{self.aggregation_type}"

        self.identifiers_path = f"{input_path}{ts_type}\\{source_type}\\{IDENTIFIERS}"
        self.times_path = f"{input_path}{ts_type}\\{source_type}\\{aggregation_type}\\{TIMES}"
        self.main_data_path = f"{input_path}{ts_type}\\{source_type}\\{aggregation_type}\\{MAIN_DATA}\\"
        self.additional_data_path = f"{input_path}{ts_type}\\{source_type}\\{aggregation_type}\\{ADDITIONAL_DATA}\\"
        self.output_path = f"{output_path}{self.file_name}"
        self.ts_info_path = f"{self.main_data_path}{TS_INFO}"

    def convert(self):

        fletcher_filter = tables.Filters(complevel=0, complib='zlib', shuffle=False, fletcher32=True)

        h5_file = None
        try:
            h5_file = tables.open_file(self.output_path, mode="w", title=self.name)
            self.create_main_group(h5_file, fletcher_filter)
            self.create_times_group(h5_file, fletcher_filter)
            self.create_additional_data_group(h5_file, fletcher_filter)
        finally:
            if h5_file is not None:
                h5_file.close()

        # print(f"Finished '{self.file_name}' file!")

    def create_main_group(self, h5_file: tables.File, fletcher_filter: tables.Filters):
        main_group = h5_file.create_group("/", self.source_type, self.source_type)
        identifiers = self.create_identifiers_table(h5_file, main_group, fletcher_filter)
        main_data_table = self.create_main_data_table(h5_file, main_group, fletcher_filter, identifiers)
        self.create_ranges_table(h5_file, main_group, main_data_table, identifiers, fletcher_filter)

    def create_identifiers_table(self, h5_file: tables.File, main_group: tables.Group, fletcher_filter: tables.Filters) -> list:
        table = h5_file.create_table(main_group, IDENTIFIERS_NAME, Identifiers, IDENTIFIERS_NAME, filters=fletcher_filter)

        df = pd.read_csv(self.identifiers_path).astype(np.uint32)
        df = df.sort_values(by=TS_ID, kind="stable")

        table.append(df[TS_ID].values)

        table.flush()
        h5_file.flush()

        return df.values

    def get_number_of_rows_of_main_data(self, identifiers: list) -> np.int64:
        number_of_rows = 0

        for ts_id, folder_id in identifiers:
            if os.path.exists(f"{self.main_data_path}{folder_id}/{ts_id}.csv"):
                df = pd.read_csv(f"{self.main_data_path}{folder_id}/{ts_id}.csv")
                number_of_rows += len(df)
            else:
                raise ValueError(f"{self.file_name} missing {ts_id} in folder {folder_id}")

        return number_of_rows

    def get_number_of_rows(self, path: str) -> int:
        return len(pd.read_csv(path))

    def get_table_descriptor(self, path: str, add_id: bool) -> tables.IsDescription:

        with open(path) as ts_info:
            columns_info = list(yaml.safe_load(ts_info).items())

            if add_id:
                columns_info.insert(0, (TS_ID, "uint32"))

            converted_columns_info = []

            for pos, column in enumerate(columns_info):
                col_name, col_type = column
                converted_columns_info.append((col_name, get_pytables_init_type(col_type, pos)))

            return dict(converted_columns_info)

        raise ValueError("Unexpected error")

    def get_column_types(self, path: str, add_id: bool) -> dict[str, np.dtype]:
        with open(path) as ts_info:
            columns_info = list(yaml.safe_load(ts_info).items())

            if add_id:
                columns_info.insert(0, (TS_ID, "uint32"))

            converted_columns_info = []

            for _, column in enumerate(columns_info):
                col_name, col_type = column
                converted_columns_info.append((col_name, get_np_type(col_type)))

            return dict(converted_columns_info)

        raise ValueError("Unexpected error")

    def create_main_data_table(self, h5_file: tables.File, main_group: tables.Group, fletcher_filter: tables.Filters, identifiers: list) -> tables.Table:
        number_of_rows = self.get_number_of_rows_of_main_data(identifiers)
        column_types = self.get_column_types(self.ts_info_path, True)
        main_data_desciptor = self.get_table_descriptor(self.ts_info_path, True)

        table = h5_file.create_table(main_group, self.main_data_table_name, main_data_desciptor, self.main_data_table_name, expectedrows=number_of_rows, filters=fletcher_filter)

        for ts_id, folder_id in identifiers:
            if os.path.exists(f"{self.main_data_path}{folder_id}/{ts_id}.csv"):
                df = pd.read_csv(f"{self.main_data_path}{folder_id}/{ts_id}.csv")

                if len(df.values) <= 0:
                    continue

                df.insert(loc=0, column=TS_ID, value=ts_id)

                for key in column_types.keys():
                    df[key] = df[key].astype(column_types[key])

                table.append(list(df.itertuples(index=False, name=None)))
                table.flush()
                del df
                gc.collect()

        table.cols.ts_id.create_csindex()
        table.flush()
        h5_file.flush()

        return table

    def create_ranges_table(self, h5_file: tables.File, main_group: tables.Group, main_data_table: tables.Table, identifiers: list, fletcher_filter: tables.Filters):
        ranges_table = h5_file.create_table(main_group, self.id_ranges_table_name, IdRanges, self.id_ranges_table_name, filters=fletcher_filter)
        row = ranges_table.row

        for ts_id, _ in identifiers:
            coordinates = main_data_table.get_where_list(f'({TS_ID} == {ts_id})')

            if (len(coordinates) == 0):
                id_range = (ts_id, 0, 0)
            else:
                id_range = (ts_id, coordinates[0], coordinates[len(coordinates) - 1] + 1)

            row[TS_ID] = id_range[0]
            row["start"] = id_range[1]
            row["end"] = id_range[2]
            row.append()

            ranges_table.flush()

        h5_file.flush()

    def create_times_group(self, h5_file: tables.File, fletcher_filter: tables.Filters):
        times_group = h5_file.create_group("/", self.times_group_name, self.times_group_name)
        times_table = h5_file.create_table(times_group, self.times_table_name, TimesDataset, self.times_table_name, filters=fletcher_filter)

        df = pd.read_csv(self.times_path)

        def nan_datetime_handler(x):
            if isinstance(x, float) or x is None:
                return -1
            else:
                return int(x)

        df["time"] = df["time"].apply(lambda x: nan_datetime_handler(x)).astype(np.int32)
        df["time_id"] = df["time_id"].astype(np.uint32)

        times_table.append(list(df.itertuples(index=False, name=None)))
        times_table.flush()

        h5_file.flush()

    def create_additional_data_group(self, h5_file: tables.File, fletcher_filter: tables.Filters):
        for additional_data_csv in Path(self.additional_data_path).glob("*.csv"):
            info_path = f"{self.additional_data_path}{additional_data_csv.stem}_info.yaml"
            number_of_rows = self.get_number_of_rows(additional_data_csv)
            column_types = self.get_column_types(info_path, True)
            descriptor = self.get_table_descriptor(info_path, True)

            table = h5_file.create_table("/", additional_data_csv.stem.lower(), descriptor, additional_data_csv.stem.lower(), expectedrows=number_of_rows, filters=fletcher_filter)

            df = pd.read_csv(additional_data_csv)
            if len(df) <= 0:
                continue

            for key in column_types.keys():
                df[key] = df[key].astype(column_types[key])

            table.append(list(df.itertuples(index=False, name=None)))
            del df
            gc.collect()

            table.flush()
            h5_file.flush()


############################################ ACTIVE PART ############################################

datasets: list[Outputer] = []

for ts_type_dir in Path(DATASET_PATH).glob("*"):
    ts_type = ts_type_dir.stem

    for source_type_dir in ts_type_dir.glob("*"):
        source_type = source_type_dir.stem

        for aggregation_type_dir in [directory for directory in source_type_dir.glob("*") if directory.is_dir()]:
            aggregation_type = aggregation_type_dir.stem

            datasets.append(Outputer(ts_type, source_type, aggregation_type, DATASET_PATH, OUTPUT_H5_PATH))

for dataset in tqdm.tqdm(datasets):
    dataset.convert()

print(f"All parts converted and are saved at {OUTPUT_H5_PATH} directory.")

############################################ ACTIVE PART ############################################
