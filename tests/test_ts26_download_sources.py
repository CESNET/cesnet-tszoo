"""CESNET-TimeSeries26 download sources: DataZoo bucket first, then the Google Drive mirrors (offline tests)."""
import hashlib
import os
from unittest import mock

import cesnet_tszoo.datasets.databases.cesnet_timeseries26.constants as c
import cesnet_tszoo.utils.download as dl
from cesnet_tszoo.datasets.databases.cesnet_timeseries26.cesnet_timeseries26 import CESNET_TimeSeries26 as TS26

NAME = "CESNET-TimeSeries26-institutions-1_day"
PAYLOAD = b"ts26-test-payload"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def test_bucket_is_first_and_shared_with_ts24():
    from cesnet_tszoo.datasets.databases.cesnet_timeseries24.cesnet_timeseries24 import CESNET_TimeSeries24
    assert TS26.bucket_url == CESNET_TimeSeries24.bucket_url
    for name in c.CHECKSUMS:
        urls = TS26.get_download_urls(name)
        assert urls[0] == f"{c.BUCKET_URL}&file={name}.h5"
        assert "drive.usercontent.google.com" in urls[1]


def test_every_file_has_a_checksum_and_a_drive_fallback():
    assert set(c.CHECKSUMS) == set(c.DOWNLOAD_URLS) == set(c.FALLBACK_DOWNLOAD_URLS)
    assert len(c.CHECKSUMS) == 30


def test_backup_drive_and_zenodo_follow_the_primary_drive(monkeypatch):
    ids = {NAME: "BACKUP_ID"}
    zen = {NAME: [dl.zenodo_file_url(1, NAME + ".h5")]}
    fallback = {n: list(u) + ([dl.google_drive_url(ids[n])] if n in ids else []) + list(zen.get(n, []))
                for n, u in c.DOWNLOAD_URLS.items()}
    monkeypatch.setattr(TS26, "fallback_download_urls", fallback)
    urls = TS26.get_download_urls(NAME)
    assert urls[0].startswith(c.BUCKET_URL)
    assert urls[1] == c.DOWNLOAD_URLS[NAME][0]
    assert "BACKUP_ID" in urls[2]
    assert "zenodo.org" in urls[3]


def _run(fake, tmp_path):
    path = os.path.join(tmp_path, NAME + ".h5")
    with mock.patch.object(dl, "resumable_download", fake):
        used = dl.download_from_sources(TS26.get_download_urls(NAME), path, expected_sha256=DIGEST, silent=True)
    return used, open(path, "rb").read()


def test_bucket_error_falls_back_to_drive(tmp_path):
    def fake(url, file_path, chunk_size=1, silent=False):
        if "datazoo" in url:
            raise ValueError("Server did not return a downloadable file (Content-Type: 'text/html').")
        open(file_path, "ab").write(PAYLOAD)
    used, data = _run(fake, tmp_path)
    assert "drive.usercontent.google.com" in used and data == PAYLOAD


def test_bucket_wrong_bytes_fall_back_to_drive(tmp_path):
    def fake(url, file_path, chunk_size=1, silent=False):
        open(file_path, "ab").write(b"not the dataset" if "datazoo" in url else PAYLOAD)
    used, data = _run(fake, tmp_path)
    assert "drive.usercontent.google.com" in used and data == PAYLOAD


def test_parallel_parts_are_assembled_in_order(tmp_path, monkeypatch):
    import random
    import time as _time
    parts = [f"https://zenodo.org/records/1/files/x.h5.{i:03d}?download=1" for i in range(1, 13)]
    payload = {u: f"<part {i}>".encode() * 1000 for i, u in enumerate(parts, start=1)}
    whole = b"".join(payload[u] for u in parts)

    def fake(url, file_path, chunk_size=1, silent=False):
        _time.sleep(random.random() / 50)  # finish in random order
        open(file_path, "wb").write(payload[url])

    monkeypatch.setattr(dl, "PARALLEL_PART_DOWNLOADS", 4)
    with mock.patch.object(dl, "resumable_download", fake):
        path = os.path.join(tmp_path, "x.h5")
        dl.download_from_sources([dl.parts_source(parts)], path, expected_sha256=hashlib.sha256(whole).hexdigest(), silent=True)
    assert open(path, "rb").read() == whole
    assert not [f for f in os.listdir(tmp_path) if f != "x.h5"]  # part files removed


def test_failed_part_moves_to_next_source(tmp_path, monkeypatch):
    parts = [f"https://zenodo.org/records/1/files/y.h5.{i:03d}?download=1" for i in range(1, 5)]

    def fake(url, file_path, chunk_size=1, silent=False):
        if "zenodo" in url and url.endswith("003?download=1"):
            raise ValueError("Server did not return a downloadable file")
        open(file_path, "ab").write(PAYLOAD if "drive" in url else b"p")

    monkeypatch.setattr(dl, "PARALLEL_PART_DOWNLOADS", 4)
    with mock.patch.object(dl, "resumable_download", fake):
        path = os.path.join(tmp_path, "y.h5")
        used = dl.download_from_sources([dl.parts_source(parts), dl.google_drive_url("ID")], path, expected_sha256=DIGEST, silent=True)
    assert "drive.usercontent.google.com" in used and open(path, "rb").read() == PAYLOAD


def test_bucket_used_when_it_serves_the_file(tmp_path):
    def fake(url, file_path, chunk_size=1, silent=False):
        open(file_path, "ab").write(PAYLOAD)
    used, data = _run(fake, tmp_path)
    assert used.startswith(c.BUCKET_URL) and data == PAYLOAD


def test_zenodo_is_the_last_source_for_every_file():
    assert set(c.ZENODO_SOURCES) == set(c.CHECKSUMS)
    for name, sources in c.ZENODO_SOURCES.items():
        urls = TS26.get_download_urls(name)
        assert urls[-len(sources):] == sources
        first = sources[0] if isinstance(sources[0], str) else sources[0]["parts"][0]
        assert first.startswith("https://zenodo.org/records/")
