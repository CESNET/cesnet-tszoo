# Adding dataset { #adding_dataset_page }

Follow these three steps to contribute a dataset to the repository.

---

## Step 1: Prepare Your Dataset Files

Your dataset must follow the directory structure below. Replace placeholder names (e.g. `<dataset_name>`, `<aggregation_1>`, `<subset_1>` ...) with your actual names - use underscores instead of spaces (e.g. `my_dataset`, `5_minutes`, `vpn` ...).

```
<dataset_name>
├─ <subset_1>
│  ├─ <source_type_1>
│  │  ├─ identifiers.csv
│  │  ├─ <aggregation_1>
│  │  │  ├─ times.csv
│  │  │  ├─ main_data
│  │  │  │  ├─ ts_info.yaml
│  │  │  │  ├─ <folder_id_1>
│  │  │  │  │  ├─ <ts_id_1>
│  │  │  │  │  │  ├─ data_1.csv
│  │  │  │  │  │  ├─ data_2.csv
│  │  │  │  │  │  ├─ ...
│  │  │  │  │  │  ├─ data_V.csv
│  │  │  │  │  │  ├─ matrix_<feature_1>_1.txt
│  │  │  │  │  │  ├─ matrix_<feature_1>_2.txt
│  │  │  │  │  │  ├─ ...
│  │  │  │  │  │  ├─ matrix_<feature_1>_E.txt
│  │  │  │  │  │  ├─ matrix_<feature_2>_1.txt
│  │  │  │  │  │  ├─ ...
│  │  │  │  │  │  └─ matrix_<feature_T>_1.txt
│  │  │  │  │  ├─ <ts_id_2>
│  │  │  │  │  ├─ ...
│  │  │  │  │  └─ <ts_id_P>
│  │  │  │  ├─ <folder_id_2>
│  │  │  │  ├─ ...
│  │  │  │  └─ <folder_id_K>
│  │  │  └─ additional_data
│  │  │     ├─ <additional_data_1>
│  │  │     │  ├─ data.csv
│  │  │     │  ├─ data_info.yaml
│  │  │     │  ├─ matrix_<feature_1>.txt
│  │  │     │  ├─ matrix_<feature_2>.txt
│  │  │     │  ├─ ...
│  │  │     │  └─ matrix_<feature_T>.txt
│  │  │     ├─ <additional_data_2>
│  │  │     ├─ ...
│  │  │     └─ <additional_data_M>
│  │  ├─ <aggregation_2>
│  │  ├─ ...
│  │  └─ <aggregation_N>
│  ├─ <source_type_2>
│  ├─ ...
│  └─ <source_type_P>
├─ <subset_2>
├─ ...
└─ <subset_R>
```

> **Naming convention:** Folder names for subsets, source types, and aggregation types are used for internal HDF5 naming. Do not use spaces - separate words with underscores (e.g. `holidays_and_weekdays`, `5_minutes`).

---

### File Specifications

#### `identifiers.csv`

Maps each time series to its folder. Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `ts_id` | integer | Unique time series ID |
| `folder_id` | integer | Folder the time series lives in |

IDs must match those used in `main_data` and `additional_data`.

---

#### `times.csv`

Maps time step IDs to UTC timestamps. Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `id_time` | integer | Unique time step ID |
| `time` | integer or null | UTC Unix epoch timestamp. Use `None` if unavailable (e.g. `5, None`) |

IDs must match those used in `main_data` and `additional_data`.

---

#### `main_data` vs `additional_data`

- Files in `main_data` must be of time series nature
- Files in `additional_data` can be of any nature
- Folders `<folder_id_1>` in `main_data` are meant to help when there is too much time series to have in one folder
    - Always use them, even when you do not need them


---

#### `<ts_id>` (one per time series, inside `main_data`)

Contains data for a single time series. The folder name must match the `ts_id` from `identifiers.csv`.

- First column in data csv file must be `id_time`, referencing IDs from `times.csv`.
- Gaps in time are permitted.
- If the file is too large, split it across multiple numbered files (`data_1.csv`, `data_2.csv`, ...) in ascending time order (lower number = earlier times).

---

#### `data_<num>.csv` or `data.csv`

- Main files for holding data
- For `main_data` files always use the numbered version even though no splitting was needed
- Always include a header row with column names.
- No spaces in column names - use underscores (e.g. `id_time`, not `id time`).

---

#### `ts_info.yaml`

Specifies the data type for each column in `main_data` CSV files, in the same order as the columns appear. Example:

```yaml
id_time:
  type: uint32
KPI:
  type: uint32
```

---

#### `data_info.yaml`

Same structure as `ts_info.yaml`, but describes the columns of `additional_data` CSV files.

---

### Supported Data Types

| Type | Variants / Notes |
|------|-----------------|
| `float` | `float16`, `float32`, `float64` |
| `int` | `int16`, `int32`, `int64` |
| `uint` | `uint16`, `uint32`, `uint64` |
| `str` | Must specify max length: `strN` (e.g. `str64`) |
| `bool` | - |
| `time` | Unix epoch UTC |

---

### Matrix Features

Some columns reference external matrix files. These must be declared in both the CSV and the YAML as a pair of entries - one for the row index ID, and one describing the matrix itself.

**Naming convention:**
- The index column in the CSV must be named `id_matrix_<feature>`.
- Corresponding matrix files must be named `matrix_<feature>_<num>.txt`.

**YAML declaration (add to both `ts_info.yaml` and `data_info.yaml` as applicable):**

```yaml
id_matrix_<feature_1>:
  type: uint32

matrix_<feature_1>:
  matrix:
    columns: 12
    rows: 12
  type: float64
```

**Rules:**

- The CSV (`data.csv` or time series data file) must include the `id_matrix_<feature>` index column for each matrix feature.
- Matrix files can also be split, following the same logic as for normal data.
- Matrices in their files are expected to be flattened by rows, where each value in row is separated by comma (i.e. one row == rows * columns)

---

## Step 2: Create Documentation

Create two Markdown files describing your dataset:

1. **Overview entry** - a short summary for inclusion in the datasets overview page. Use `./docs/datasets_overview.md` as a reference for style and format.

2. **Standalone dataset page** - a full description of your dataset. Use `./docs/cesnet_timeseries24.md` as a reference for style and format.

---

## Step 3: Open an Issue

Create a GitHub issue that includes:

- Any other relevant information about the dataset.
- A download link for the dataset files. Besides the CESNET DataZoo `bucket_url`, a database can list backup mirrors
  in `fallback_download_urls` (`dataset file name -> [source, ...]`, tried in order after the bucket) and the SHA-256
  of every file in `checksums`; a source that fails or returns a file with another digest is skipped. Build the URLs
  with `cesnet_tszoo.utils.download.google_drive_url(file_id)` and `zenodo_file_url(record_id, file_name)`. A file too
  large for one Zenodo record (50 GB) can be published as consecutive parts (`scripts/zenodo_pack.py` splits the
  files, computes the checksums and plans the records); the source is then `parts_source([url_of_part_1, ...])` and
  the library downloads the parts in parallel, re-assembles the file and verifies it. See
  `datasets/databases/cesnet_agg23/constants.py` and `datasets/databases/cesnet_timeseries26/constants.py`.
- Links to both Markdown documentation files created in Step 2.
