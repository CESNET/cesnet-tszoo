import logging

import pandas as pd
import numpy as np

from cesnet_tszoo.utils.enums import AnnotationType


class Annotations():
    """Used as wrapper around dictionaries used for storing annotations data."""

    def __init__(self):
        self.time_series_annotations = {}
        self.time_annotations = {}
        self.time_in_series_annotations = {}

        self.logger = logging.getLogger("annotations")

    def add_annotation(self, annotation: str, annotation_group: str, ts_id: int | None, id_time: int | None) -> None:
        """
        Add an annotation to the specified `annotation_group` identified by `ts_id` and `id_time`.

        If the `annotation_group` does not already exist, it will be created.
        """

        assert ts_id is not None or id_time is not None, "At one of the ids must be set."
        assert annotation_group is not None, "Annotation group must be set."

        if ts_id is not None and id_time is not None:
            self.add_annotation_group(annotation_group, AnnotationType.BOTH, True)
            self.time_in_series_annotations[annotation_group][(ts_id, id_time)] = annotation
            self.logger.debug("Annotation added to annotation_group: %s in %s.", annotation_group, AnnotationType.BOTH)
        elif id_time is not None:
            self.add_annotation_group(annotation_group, AnnotationType.ID_TIME, True)
            self.time_annotations[annotation_group][id_time] = annotation
            self.logger.debug("Annotation added to annotation_group: %s in %s.", annotation_group, AnnotationType.ID_TIME)
        elif ts_id is not None:
            self.add_annotation_group(annotation_group, AnnotationType.TS_ID, True)
            self.time_series_annotations[annotation_group][ts_id] = annotation
            self.logger.debug("Annotation added to annotation_group: %s in %s.", annotation_group, AnnotationType.TS_ID)
        else:
            raise NotImplementedError("Should not happen.")

    def clear_time_in_time_series(self):
        """Clears `time_in_series_annotations` dictionary. """

        self.time_in_series_annotations = {}
        self.logger.debug("Cleared time_in_series_annotations.")

    def clear_time(self):
        """Clears `time_annotations` dictionary. """

        self.time_annotations = {}
        self.logger.debug("Cleared time_annotations.")

    def clear_time_series(self):
        """Clears `time_series_annotations` dictionary. """

        self.time_series_annotations = {}
        self.logger.debug("Cleared time_series_annotations.")

    def remove_annotation(self, annotation_group: str, ts_id: int | None, id_time: int | None, silent: bool = False):
        """ Removes annotation from `annotation_group` and `ts_id`, `id_time`. """

        assert ts_id is not None or id_time is not None, "At one of the ids must be set."
        assert annotation_group is not None, "Annotation group must be set."

        if ts_id is not None and id_time is not None:
            assert annotation_group in self.time_in_series_annotations, "Annotation group must exist in time_in_series_annotations."
            if (ts_id, id_time) not in self.time_in_series_annotations[annotation_group]:
                if not silent:
                    self.logger.info("Annotation does not exist.")

                return

            self.time_in_series_annotations[annotation_group].pop((ts_id, id_time))

        elif id_time is not None:
            assert annotation_group in self.time_annotations, "Annotation group must exist in time_annotations."
            if id_time not in self.time_annotations[annotation_group]:
                if not silent:
                    self.logger.info("Annotation does not exist.")

                return

            self.time_annotations[annotation_group].pop(id_time)

        elif ts_id is not None:
            assert annotation_group in self.time_series_annotations, "Annotation group must exist in time_series_annotations."
            if ts_id not in self.time_series_annotations[annotation_group]:
                if not silent:
                    self.logger.info("Annotation does not exist.")

                return

            self.time_series_annotations[annotation_group].pop(ts_id)

    def add_annotation_group(self, annotation_group: str, on: AnnotationType, silent: bool = False):
        """ Adds `annotation_group` to dictionary based on `on`. """

        if on == AnnotationType.TS_ID:
            if annotation_group in self.time_series_annotations:
                if not silent:
                    self.logger.info("Annotation group %s already exists.", annotation_group)

                return

            self.time_series_annotations[annotation_group] = {}
        elif on == AnnotationType.ID_TIME:
            if annotation_group in self.time_annotations:
                if not silent:
                    self.logger.info("Annotation group %s already exists.", annotation_group)

                return

            self.time_annotations[annotation_group] = {}
        elif on == AnnotationType.BOTH:
            if annotation_group in self.time_in_series_annotations:
                if not silent:
                    self.logger.info("Annotation group %s already exists.", annotation_group)

                return

            self.time_in_series_annotations[annotation_group] = {}

    def remove_annotation_group(self, annotation_group: str, on: AnnotationType, silent: bool = False):
        """ Removes `annotation_group` from dictionary based on `on`. """

        if on == AnnotationType.TS_ID:
            if annotation_group not in self.time_series_annotations:
                if not silent:
                    self.logger.info("Annotation group %s does not exist.", annotation_group)

                return

            self.time_series_annotations.pop(annotation_group)
        elif on == AnnotationType.ID_TIME:
            if annotation_group not in self.time_annotations:
                if not silent:
                    self.logger.info("Annotation group %s does not exist.", annotation_group)

                return

            self.time_annotations.pop(annotation_group)
        elif on == AnnotationType.BOTH:
            if annotation_group not in self.time_in_series_annotations:
                if not silent:
                    self.logger.info("Annotation group %s does not exist.", annotation_group)

                return

            self.time_in_series_annotations.pop(annotation_group)

    def get_annotations(self, on: AnnotationType, ts_id_name: str) -> pd.DataFrame:
        """Returns annotations based on `on` as Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). """

        rows = {}
        columns = []
        annotations = None
        point_size = None

        if on == AnnotationType.TS_ID:
            columns.append(ts_id_name)
            annotations = self.time_series_annotations
            point_size = 1

        elif on == AnnotationType.ID_TIME:
            columns.append("id_time")
            annotations = self.time_annotations
            point_size = 1

        elif on == AnnotationType.BOTH:
            columns.append(ts_id_name)
            columns.append("id_time")
            annotations = self.time_in_series_annotations
            point_size = 2

        for i, group in enumerate(annotations):
            columns.append(group)
            i = i + point_size  # offset for ts_id and id_time
            for point in annotations[group]:
                if point in rows:
                    row = rows[point]
                else:
                    row = np.array([None for _ in range(len(annotations) + point_size)])
                    row[:point_size] = point
                    rows[point] = row

                row[i] = annotations[group][point]

        return pd.DataFrame(rows.values(), columns=columns)
