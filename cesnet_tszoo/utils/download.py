import gzip
import hashlib
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from cesnet_tszoo.utils.enums import DownloadSource

ISSUES_URL = "https://github.com/CESNET/cesnet-tszoo/issues"

# (connect, read) timeouts in seconds; a source that stalls raises requests.exceptions.Timeout, so
# download_from_sources moves on to the next mirror instead of waiting forever.
REQUEST_TIMEOUT = (30, 300)

# Number of parts of a multi-part source (see parts_source) downloaded at the same time.
# Override with the environment variable CESNET_TSZOO_PARALLEL_DOWNLOADS (1 = sequential).
PARALLEL_PART_DOWNLOADS = int(os.environ.get("CESNET_TSZOO_PARALLEL_DOWNLOADS", "4"))


def simple_download(url: str, file_path: str):
    r = requests.get(url)
    with open(file_path, "wb") as file:
        file.write(r.content)


def google_drive_url(file_id: str) -> str:
    """
    Build a direct-download URL for a file shared on Google Drive ("Anyone with the link").

    `file_id` is the part of the sharing link between `/d/` and `/view`
    (`https://drive.google.com/file/d/<file_id>/view`). `confirm=t` skips the
    "file too large for virus scan" interstitial page, so the same URL works for files above 100 MB.
    The host answers with `Content-Length`, `Accept-Ranges: bytes` and honors `Range`, so
    [`resumable_download`][cesnet_tszoo.utils.download.resumable_download] can resume partial files.
    """

    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def zenodo_file_url(record_id: int | str, file_name: str) -> str:
    """
    Build the direct-download URL of a file in a Zenodo record (`https://zenodo.org/records/<record_id>/files/<file_name>?download=1`).

    Zenodo answers with `Content-Length` and honors `Range`, so partial downloads are resumed.
    """

    return f"https://zenodo.org/records/{record_id}/files/{file_name}?download=1"


def parts_source(part_urls: list[str], gzip_parts: bool = False) -> dict:
    """
    Describe a dataset file that is published as several consecutive parts (e.g. to fit the per-record size limit of Zenodo).

    The parts are downloaded one by one and concatenated in the given order; when `gzip_parts` is True every part is
    a gzip member and is decompressed while concatenating. The resulting file is verified against the checksum of
    the original (whole, uncompressed) file. Use the returned dict as one entry of `download_urls` next to plain URLs.
    """

    return {"parts": list(part_urls), "gzip": gzip_parts}


def source_kind(source: str | dict) -> DownloadSource | None:
    """Returns the [`DownloadSource`][cesnet_tszoo.utils.enums.DownloadSource] a download source belongs to (by its host), or None for other hosts."""

    url = source if isinstance(source, str) else source["parts"][0]
    host = urlparse(url).netloc.lower()
    if host == "liberouter.org" or host.endswith(".liberouter.org"):
        return DownloadSource.DATAZOO
    if host in ("drive.google.com", "drive.usercontent.google.com", "docs.google.com"):
        return DownloadSource.GOOGLE_DRIVE
    if host == "zenodo.org" or host.endswith(".zenodo.org"):
        return DownloadSource.ZENODO
    return None


def sha256_of_file(file_path: str, chunk_size: int = 1024**2) -> str:
    """Returns hex SHA-256 digest of the file at `file_path`. """

    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resumable_download(url: str, file_path: str, chunk_size: int = 1024**2, silent: bool = False):
    """
    Downloads `url` to `file_path`, resuming from the already present bytes when `file_path` exists.

    Raises `requests.exceptions.HTTPError` when the server answers with an error status and
    `ValueError` when the server does not return a binary file (e.g. an HTML page instead of the dataset).
    """

    r1 = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    r1.raise_for_status()

    content_type = r1.headers.get("Content-Type", "")
    if "text/html" in content_type or "Content-Length" not in r1.headers:
        r1.close()
        raise ValueError(f"Server at {url} did not return a downloadable file (Content-Type: '{content_type}').")

    redirected_url = r1.url
    content_size = int(r1.headers["Content-Length"])
    r1.close()

    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)
    else:
        temp_size = 0
    if temp_size == content_size:
        return
    if temp_size > content_size:
        # Leftover from a different (older/bigger) file - start over.
        os.remove(file_path)
        temp_size = 0

    headers = {"Range": f"bytes={temp_size}-"}
    r2 = requests.get(redirected_url, stream=True, headers=headers, timeout=REQUEST_TIMEOUT)
    r2.raise_for_status()
    if "text/html" in r2.headers.get("Content-Type", ""):
        # The second request can still hit an interstitial page (e.g. Google Drive "quota exceeded") even when
        # the first one advertised the file; never write it into the dataset file.
        r2.close()
        raise ValueError(f"Server at {url} did not return a downloadable file (Content-Type: '{r2.headers.get('Content-Type', '')}').")
    if temp_size > 0 and r2.status_code != 206:
        # Server ignored the Range header and sends the whole file again.
        os.remove(file_path)
        temp_size = 0

    if not silent:
        print(f"File size: {content_size / (1024**3):0.2f}GB")
        print(f"Remaining: {(content_size - temp_size) / (1024**3):0.2f}GB")

    progress_bar = tqdm(total=content_size - temp_size, unit="B", unit_scale=True, unit_divisor=1024, disable=silent)
    with open(file_path, "ab") as file:
        for data in r2.iter_content(chunk_size=chunk_size):
            file.write(data)
            progress_bar.update(len(data))
    progress_bar.close()

    downloaded_size = os.path.getsize(file_path)
    if downloaded_size != content_size:
        # Truncated or replaced payload - leave the partial file for a later resume only when it is shorter.
        if downloaded_size > content_size:
            os.remove(file_path)
        raise ValueError(f"Download from {url} is incomplete ({downloaded_size} of {content_size} bytes).")


def download_parts(part_urls: list[str], file_path: str, gzip_parts: bool = False, chunk_size: int = 1024**2, silent: bool = False):
    """
    Downloads the consecutive parts of one file (see [`parts_source`][cesnet_tszoo.utils.download.parts_source]) and assembles them into `file_path`.

    Every part is downloaded with [`resumable_download`][cesnet_tszoo.utils.download.resumable_download] to
    `<file_path>.<index>` (resumed on the next call), then the parts are concatenated - decompressed when
    `gzip_parts` is True - into `file_path` and deleted.
    """

    logger = logging.getLogger("wrapper_dataset")
    part_files = [f"{file_path}.{i:03d}" for i in range(1, len(part_urls) + 1)]
    n = len(part_urls)
    workers = max(1, min(PARALLEL_PART_DOWNLOADS, n))

    if workers == 1:
        for i, (url, part_file) in enumerate(zip(part_urls, part_files), start=1):
            if not silent:
                print(f"Part {i}/{n}")
            logger.info("Downloading part %d/%d from %s", i, n, url)
            resumable_download(url=url, file_path=part_file, chunk_size=chunk_size, silent=silent)
    else:
        # Several parts at once: a single connection to a repository such as Zenodo is often slow, so parallel
        # parts multiply the throughput. Every part is resumable on its own; per-part progress bars are disabled.
        done = []

        def fetch(item):
            i, url, part_file = item
            logger.info("Downloading part %d/%d from %s", i, n, url)
            resumable_download(url=url, file_path=part_file, chunk_size=chunk_size, silent=True)
            done.append(i)
            if not silent:
                print(f"Part {i}/{n} done ({len(done)}/{n})")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, (i, url, part_file)) for i, (url, part_file) in enumerate(zip(part_urls, part_files), start=1)]
            for future in futures:
                future.result()  # re-raise the first failure; finished parts stay on disk for a resume

    logger.info("Assembling %d parts into %s", len(part_files), file_path)
    with open(file_path, "wb") as output:
        for part_file in part_files:
            opener = gzip.open if gzip_parts else open
            with opener(part_file, "rb") as part:
                shutil.copyfileobj(part, output, chunk_size)

    for part_file in part_files:
        os.remove(part_file)


def download_from_sources(urls: list[str | dict], file_path: str, expected_sha256: str | None = None, chunk_size: int = 1024**2, silent: bool = False) -> str:
    """
    Downloads the file from the first working source in `urls` (mirrors are tried in order).

    A source is either a plain URL or a multi-part description made by
    [`parts_source`][cesnet_tszoo.utils.download.parts_source]. A mirror is skipped when it is unreachable,
    answers with an error status, or returns something that is not a binary file. When `expected_sha256` is
    given, the downloaded file is verified and a mismatching file is deleted before the next mirror is tried.
    The data are written to `<file_path>.part` and renamed to `file_path` only after the download (and
    verification) succeeded, so an interrupted download never leaves a truncated file that would be mistaken
    for a complete dataset; the `.part` file (or the downloaded parts) is resumed on the next call.

    Returns:
        The URL (or the URL of the first part) the file was finally downloaded from.

    Raises:
        RuntimeError: when no mirror provided the file.
    """

    logger = logging.getLogger("wrapper_dataset")

    if len(urls) == 0:
        raise RuntimeError(f"No download source is configured for {os.path.basename(file_path)}.")

    errors = []
    part_path = file_path + ".part"
    for source in urls:
        url = source if isinstance(source, str) else source["parts"][0]
        try:
            if isinstance(source, str):
                logger.info("Downloading from %s", url)
                resumable_download(url=url, file_path=part_path, chunk_size=chunk_size, silent=silent)
            else:
                logger.info("Downloading %d parts, first from %s", len(source["parts"]), url)
                download_parts(part_urls=source["parts"], file_path=part_path, gzip_parts=source.get("gzip", False), chunk_size=chunk_size, silent=silent)

            if expected_sha256 is not None:
                actual_sha256 = sha256_of_file(part_path)
                if actual_sha256.lower() != expected_sha256.lower():
                    os.remove(part_path)
                    raise ValueError(f"SHA-256 mismatch (expected {expected_sha256}, got {actual_sha256}).")

            os.replace(part_path, file_path)
            return url
        except (requests.exceptions.RequestException, ValueError, OSError) as e:
            logger.warning("Download from %s failed: %s", url, e)
            errors.append(f"{url}: {e}")

    raise RuntimeError("Could not download the dataset from any of the configured sources:\n  "
                       + "\n  ".join(errors)
                       + f"\nPlease contact us at {ISSUES_URL}.")
