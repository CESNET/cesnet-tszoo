# Annotations

This tutorial will look at how to use annotations.

!!! info "Note"
    For every option and more detailed examples refer to Jupyter notebook [`annotations`](https://github.com/CESNET/cesnet-tszoo/blob/main/tutorial_notebooks/annotations.ipynb).

## Basics

- You can get annotations for specific type with `get_annotations` method. 
- Method `get_annotations` returns annotations as Pandas Dataframe.

There are three annotation types:

1. **AnnotationType.TS_ID** -> Annotations for whole specific time series
2. **AnnotationType.ID_TIME** -> Annotations for specific time... independent on time series
3. **AnnotationType.BOTH** -> Annotations for specific time in specific time series

```python

from cesnet_tszoo.utils.enums import AnnotationType                                                                          

dataset.get_annotations(on=AnnotationType.TS_ID)
dataset.get_annotations(on=AnnotationType.ID_TIME)
dataset.get_annotations(on=AnnotationType.BOTH)

```

## Annotation groups
- Annotation group could be understood as column names in Dataframe/CSV.
- You can add annotation groups or remove them.

```python

from cesnet_tszoo.utils.enums import AnnotationType                                                                          

# Adding groups
dataset.add_annotation_group(annotation_group="test1", on=AnnotationType.TS_ID)
dataset.add_annotation_group(annotation_group="test2", on=AnnotationType.ID_TIME)
dataset.add_annotation_group(annotation_group="test3", on=AnnotationType.BOTH)

# Removing groups
dataset.remove_annotation_group(annotation_group="test1", on=AnnotationType.TS_ID)
dataset.remove_annotation_group(annotation_group="test2", on=AnnotationType.ID_TIME)
dataset.remove_annotation_group(annotation_group="test3", on=AnnotationType.BOTH)

```

## Annotation values
- Annotations are specific values for selected annotation group and AnnotationType.
- You can add annotations or remove them.
- Adding annotation
    - When adding annotation to annotation group that does not exist, it will be created.
    - To override existing annotation, you just need to specify same `annotation_group`, `ts_id`, `id_time` and new annotation.
    - Setting `enforce_ids` to True, ensures that inputted `ts_id` and `id_time` must belong to used dataset.
- Removing annotations
    - Removing annotation from every annotation group of a row, removes that row from Dataframe.

```python                                                                     

# Adding annotations
dataset.add_annotation(annotation="test_annotation1_3", annotation_group="test1", ts_id=3, id_time=None, enforce_ids=True) # Adds to AnnotationType.TS_ID
dataset.add_annotation(annotation="test_annotation2_0", annotation_group="test2", ts_id=None, id_time=0, enforce_ids=True) # Adds to AnnotationType.ID_TIME
dataset.add_annotation(annotation="test_annotation3_3_0", annotation_group="test3", ts_id=3, id_time=0, enforce_ids=True) # Adds to AnnotationType.BOTH

# Removing annotations
dataset.remove_annotation(annotation_group="test1", ts_id=3, id_time=None) # Removes from AnnotationType.TS_ID
dataset.remove_annotation(annotation_group="test2", ts_id=None, id_time=0 ) # Removes from AnnotationType.ID_TIME
dataset.remove_annotation(annotation_group="test3", ts_id=3, id_time=0 ) # Removes from AnnotationType.BOTH

```    

## Exporting annotations
- You can export your created annotation with `save_annotations` method.
- `save_annotations` creates CSV file at: `os.path.join(dataset.metadata.annotations_root, identifier)`.
- When parameter `force_write` is True, existing files with same name will be overwritten.
- You should not add ".csv" to identifier, because it will be added automatically.

```python                                                                     

from cesnet_tszoo.utils.enums import AnnotationType   

dataset.save_annotations(identifier="test_name", on=AnnotationType.BOTH, force_write=True)

```   

## Importing annotations
- You can import already existing annotations, be it your own or already built-in one.
- Setting `enforce_ids` to True, ensures that all `ts_id` or `id_time` from imported annotations must belong to used dataset.
- Method `import_annotations` automatically detects what AnnotationType imported annotations is, based on existing ts_id (expects name of ts_id for used dataset) or id_time columns.
- First, it attempts to load the built-in annotations, if no built-in annotations with such an identifier exists, it attempts to load a custom annotations from the `"data_root"/tszoo/annotations/` directory.

```python                                                                     

from cesnet_tszoo.utils.enums import AnnotationType   

dataset.import_annotations(identifier="test_name", enforce_ids=True)

```   

