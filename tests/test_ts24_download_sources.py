"""CESNET-TimeSeries24 download sources: DataZoo bucket first, Zenodo as the fallback (offline tests)."""
import cesnet_tszoo.datasets.databases.cesnet_timeseries24.constants as c
from cesnet_tszoo.datasets.databases.cesnet_timeseries24.cesnet_timeseries24 import CESNET_TimeSeries24 as TS24


def test_every_file_has_a_checksum_and_a_zenodo_source():
    assert set(c.CHECKSUMS) == set(c.ZENODO_SOURCES)
    assert len(c.CHECKSUMS) == 12
    assert TS24.checksums is c.CHECKSUMS


def test_bucket_first_then_zenodo():
    for name, sources in c.ZENODO_SOURCES.items():
        urls = TS24.get_download_urls(name)
        assert urls[0] == f"{TS24.bucket_url}&file={name}.h5"
        assert urls[1:] == sources
        first = sources[0] if isinstance(sources[0], str) else sources[0]["parts"][0]
        assert first.startswith("https://zenodo.org/records/")


def test_dataset_names_match_the_library_naming():
    # TS24 uses the backward-compatible names (minutes/hour/day), which are also the DataZoo and Zenodo names.
    for name in c.CHECKSUMS:
        assert name.rsplit("-", 1)[1] in {"minutes", "hour", "day"}
