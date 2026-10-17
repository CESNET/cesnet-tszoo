"""Choosing the download source (`download_source` of `get_dataset`): auto, datazoo, google_drive, zenodo (offline tests)."""
import inspect

import pytest

from cesnet_tszoo.datasets.databases.cesnet_database import CesnetDatabase
from cesnet_tszoo.utils.download import google_drive_url, parts_source, source_kind, zenodo_file_url
from cesnet_tszoo.utils.enums import DownloadSource


class _DB(CesnetDatabase):
    name = "Test-DB"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=test"
    fallback_download_urls = {"Test-DB-a": [google_drive_url("ID1"), google_drive_url("ID2"),
                                            parts_source([zenodo_file_url(1, "a.h5.001"), zenodo_file_url(1, "a.h5.002")])]}


def test_source_kind_by_host():
    assert source_kind(f"{_DB.bucket_url}&file=x.h5") is DownloadSource.DATAZOO
    assert source_kind(google_drive_url("X")) is DownloadSource.GOOGLE_DRIVE
    assert source_kind(zenodo_file_url(1, "x.h5")) is DownloadSource.ZENODO
    assert source_kind(parts_source([zenodo_file_url(1, "x.h5.001")])) is DownloadSource.ZENODO
    assert source_kind("https://example.org/x.h5") is None


def test_auto_keeps_every_source_in_order():
    urls = _DB.get_download_urls("Test-DB-a")
    assert [source_kind(u) for u in urls] == [DownloadSource.DATAZOO, DownloadSource.GOOGLE_DRIVE, DownloadSource.GOOGLE_DRIVE, DownloadSource.ZENODO]
    assert urls == _DB.get_download_urls("Test-DB-a", "auto") == _DB.get_download_urls("Test-DB-a", DownloadSource.AUTO)


@pytest.mark.parametrize("choice, count", [("datazoo", 1), ("google_drive", 2), ("zenodo", 1), (DownloadSource.ZENODO, 1)])
def test_single_source_is_selected(choice, count):
    urls = _DB.get_download_urls("Test-DB-a", choice)
    assert len(urls) == count and all(source_kind(u) is DownloadSource(choice) for u in urls)


def test_missing_source_raises_with_the_available_ones():
    class _OnlyBucket(_DB):
        fallback_download_urls = {}
    with pytest.raises(ValueError, match=r"no 'zenodo' download source.*\['datazoo'\]"):
        _OnlyBucket.get_download_urls("Test-DB-a", "zenodo")


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        _DB.get_download_urls("Test-DB-a", "dropbox")


def test_every_database_exposes_download_source():
    import importlib
    import pkgutil
    import cesnet_tszoo.datasets.databases as databases
    classes = set()
    for module in pkgutil.walk_packages(databases.__path__, databases.__name__ + "."):
        for value in vars(importlib.import_module(module.name)).values():
            if isinstance(value, type) and issubclass(value, CesnetDatabase) and "get_dataset" in value.__dict__:
                classes.add(value)
    assert len(classes) >= 7
    for cls in classes:
        parameters = inspect.signature(cls.get_dataset).parameters
        assert list(parameters)[-1] == "download_source", cls
        assert parameters["download_source"].default == "auto", cls
