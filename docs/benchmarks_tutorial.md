# Benchmarks { #benchmark_tutorial }

This tutorial will look at how to use benchmarks.

Only time-based will be used, because all methods work almost the same way for other dataset types.

!!! info "Note"
    For every option and more detailed examples refer to Jupyter notebook [`benchmarks`](https://github.com/CESNET/cesnet-ts-zoo-tutorials/blob/main/benchmarks.ipynb)

Benchmarks can consist of various parts:

- identifier of used config
- identifier of used annotations (for each AnnotationType)
- identifier of related_results (only available for premade benchmarks)
- Used SourceType and AggregationType
- Database name (here it would be CESNET_TimeSeries24)
- Whether config or annotations are built-in

## Importing benchmarks

- You can import your own or built-in benchmark with `load_benchmark` function.
- When importing benchmark with annotations that exist, but are not downloaded, they will be downloaded (only works for built-in annotations),
- First, it attempts to load the built-in benchmark, if no built-in benchmark with such an identifier exists, it attempts to load a custom benchmark from the `"data_root"/tszoo/benchmarks/` directory.

```python

from cesnet_tszoo.benchmarks import load_benchmark                                                                       

# Imports built-in benchmark
# Can get related_results with `get_related_results` method.
# Method `get_related_results` returns pandas Dataframe. 
benchmark = load_benchmark(identifier="2e92831cb502", data_root="/some_directory/")
dataset = benchmark.get_initialized_dataset(display_config_details=True, check_errors=False, workers="config")

# Imports custom benchmark
# Looks for benchmark at: `os.path.join("/some_directory/", "tszoo", "benchmarks", identifier)`
benchmark = load_benchmark(identifier="test2", data_root="/some_directory/")
dataset = benchmark.get_initialized_dataset(display_config_details=True, check_errors=False, workers="config")

```

## Exporting benchmarks

- You can use method `save_benchmark` to save benchmark.
- Saving benchmark creates YAML file, which hold metadata, at: `os.path.join(dataset.metadata.benchmarks_root, identifier)`.
- Saving benchmark automatically creates files for config and annotations with identifiers matching benchmark identifier
  - config will be saved at: `os.path.join(dataset.metadata.configs_root, identifier)`
  - annotations will be saved at: `os.path.join(dataset.metadata.annotations_root, identifier, str(AnnotationType))`
  - When parameter `force_write` is True, existing files with the same name will be overwritten.
- When using imported config or annotations, only their identifier will be passed to benchmark and no new files will get created
  - if calling anything that changes annotations, it will no longer be taken as imported
- Only annotations with at least one value will be exported.
- You can export benchmarks with custom transformers or fillers, but should share their source code along with benchmark

```python

from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.configs import TimeBasedConfig                                                                            

time_based_dataset = CESNET_TimeSeries24.get_dataset(data_root="/some_directory/", source_type=SourceType.IP_ADDRESSES_FULL, aggregation=AgreggationType.AGG_1_DAY, dataset_type=DatasetType.TIME_BASED, display_details=True)
config = TimeBasedConfig([1548925, 443967], train_time_period=1.0, features_to_take=["n_flows", "n_packets", "n_bytes"], transform_with=None)

# Call on time-based dataset to use created config -> must be done before saving exporting benchmark
time_based_dataset.set_dataset_config_and_initialize(config, workers=0, display_config_details=True)

time_based_dataset.save_benchmark(identifier="test1", force_write=True)

```

## Other
Instead of exporting or importing whole benchmark you can do for specific config or annotations.

### Config
- Saving config
    - When parameter `force_write` is True, existing files with the same name will be overwritten.
    - Config will be saved as pickle file at: `os.path.join(dataset.metadata.configs_root, identifier)`
    - When parameter `create_with_details_file` is True, text file with config details will be exported along pickle config.
- Importing config
    - - First, it attempts to load the built-in config, if no built-in config with such an identifier exists, it attempts to load a custom config from the `"data_root"/tszoo/configs/` directory.

```python

from cesnet_tszoo.configs import TimeBasedConfig                                                                      

config = TimeBasedConfig([1548925, 443967], train_time_period=1.0, features_to_take=["n_flows", "n_packets", "n_bytes"], transform_with=None)

time_based_dataset.set_dataset_config_and_initialize(config, workers=0, display_config_details=True)

# Exports config
time_based_dataset.save_config(identifier="test_config1", create_with_details_file=True, force_write=True)

# Imports custom config
time_based_dataset.import_config(identifier="test_config1", display_config_details=True, workers="config")

```

### Annotations
- Saving annotation
    - When parameter `force_write` is True, existing files with the same name will be overwritten.
    - Annotations will be saved as CSV file at: `os.path.join(dataset.metadata.annotations_root, identifier)`.
- Importing annotation
    - First, it attempts to load the built-in annotations, if no built-in annotations with such an identifier exists, it attempts to load a custom annotations from the `"data_root"/tszoo/annotations/` directory.

```python

from cesnet_tszoo.utils.enums import AnnotationType                                                                    

dataset.add_annotation(annotation="test_annotation3_3_0", annotation_group="test3", ts_id=3, id_time=0, enforce_ids=True)
dataset.add_annotation(annotation="test_annotation3_3_5", annotation_group="test3_2", ts_id=3, id_time=5, enforce_ids=True)
dataset.add_annotation(annotation="test_annotation3_5_0", annotation_group="test3", ts_id=5, id_time=0, enforce_ids=True)
dataset.add_annotation(annotation="test_annotation3_5_1", annotation_group="test3_2", ts_id=5, id_time=1, enforce_ids=True)
dataset.get_annotations(on=AnnotationType.BOTH)

# Exports annotation of type BOTH
dataset.save_annotations(identifier="test_annotations1", on=AnnotationType.BOTH, force_write=True)

# Imports custom annotations
dataset.import_annotations(identifier="test_annotations1", enforce_ids=True)

```