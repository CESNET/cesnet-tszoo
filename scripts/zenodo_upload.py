#!/usr/bin/env python3
"""
Upload one record of a `zenodo_pack.py` package to Zenodo as a DRAFT (never publishes).

For record <k> of the manifest: create a new draft deposition (or reuse `--deposition <id>`), reserve its DOI,
upload every piece of the record plus SHA256SUMS.txt and optional extra files (README.md, LICENSE.txt), set the
metadata from a JSON file, and verify that the MD5 Zenodo reports for every uploaded file matches the local file.
Publishing stays a manual step in the Zenodo web interface (a published record can no longer be changed).

The token is read from ~/.zenodo_token (scopes deposit:write, deposit:actions) or the ZENODO_TOKEN variable.
Symbolic links in the record directory are followed (the target file is uploaded).

Example:
  python scripts/zenodo_upload.py --manifest /Volumes/NewDisk/tszoo-mirror/zenodo/CESNET-TimeSeries26/pack/manifest.json \\
      --record 1 --metadata zenodo_ts26_record1.json \\
      --extra /Volumes/NewDisk/tszoo-mirror/zenodo/CESNET-TimeSeries26/README.md
  # add --sandbox to try it on https://sandbox.zenodo.org first (needs a sandbox token)
"""

import argparse
import hashlib
import json
import os
import sys
import time

import requests

CHUNK = 64 * 1024**2


def token() -> str:
    path = os.path.expanduser("~/.zenodo_token")
    tok = os.environ.get("ZENODO_TOKEN") or (open(path).read().strip() if os.path.exists(path) else "")
    if not tok:
        sys.exit("No token: put it into ~/.zenodo_token (chmod 600) or set ZENODO_TOKEN.")
    return tok


def md5_of(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProgressReader:
    """File object that reports progress while requests streams it."""

    def __init__(self, path: str):
        self.f = open(path, "rb")
        self.size = os.path.getsize(path)
        self.done = 0
        self.t0 = time.time()
        self.name = os.path.basename(path)
        self.last = 0.0

    def __len__(self):
        return self.size

    def read(self, n=-1):
        data = self.f.read(CHUNK if n is None or n < 0 else n)
        self.done += len(data)
        if time.time() - self.last > 30 or self.done == self.size:
            self.last = time.time()
            rate = self.done / max(1e-6, time.time() - self.t0) / 1e6
            print(f"  {self.name}: {self.done / 1e9:6.2f} / {self.size / 1e9:.2f} GB ({rate:.0f} MB/s)", flush=True)
        return data


def upload_legacy(s, api, dep_id, bucket, paths, names):
    """Old deposit API: one PUT per file into the deposition bucket."""
    existing = {f["filename"]: f for f in s.get(f"{api}/deposit/depositions/{dep_id}/files", timeout=60).json()}
    for path, name in zip(paths, names):
        local_md5 = md5_of(path)
        if name in existing and existing[name].get("checksum") == local_md5:
            print(f"  {name}: already uploaded, MD5 matches")
            continue
        for attempt in range(1, 4):
            try:
                r = s.put(f"{bucket}/{name}", data=ProgressReader(path), timeout=(60, 3600))
                r.raise_for_status()
                remote = r.json().get("checksum", "").replace("md5:", "")
                if remote != local_md5:
                    raise ValueError(f"MD5 mismatch: local {local_md5}, Zenodo {remote}")
                print(f"  {name}: uploaded, MD5 verified")
                break
            except (requests.RequestException, ValueError) as e:
                print(f"  {name}: attempt {attempt} failed: {e}", flush=True)
                if attempt == 3:
                    sys.exit(f"giving up on {name}; rerun with --deposition {dep_id} to resume")
                time.sleep(30 * attempt)


def set_title_description(s, api, rec_id, describe_dir, k):
    """Replace only metadata.title and metadata.description of the draft (InvenioRDM draft API)."""
    title = open(os.path.join(describe_dir, f"record_{k}_title.txt")).read().strip()
    desc = open(os.path.join(describe_dir, f"record_{k}_description.html")).read().strip()
    url = f"{api}/records/{rec_id}/draft"
    r = s.get(url, timeout=60)
    r.raise_for_status()
    draft = r.json()
    body = {key: draft[key] for key in ("access", "files", "metadata", "pids", "custom_fields") if key in draft}
    body["metadata"]["title"] = title
    body["metadata"]["description"] = desc
    r = s.put(url, json=body, timeout=120)
    if not r.ok:
        print(f"WARNING: title/description not set ({r.status_code}): {r.text[:500]}; files are uploaded anyway", flush=True)
        return
    print(f"title and description set ({len(desc)} characters)", flush=True)


def prune_rdm(s, api, rec_id, keep):
    """Delete draft files that do not belong to this record."""
    r = s.get(f"{api}/records/{rec_id}/draft/files", timeout=60)
    r.raise_for_status()
    for e in r.json().get("entries", []):
        if e["key"] not in keep:
            s.delete(f"{api}/records/{rec_id}/draft/files/{e['key']}", timeout=60).raise_for_status()
            print(f"  removed from draft (not in this record's plan): {e['key']}", flush=True)


def upload_rdm(s, api, rec_id, paths, names, tries, parallel=1):
    """InvenioRDM draft-files API (what the Zenodo web uploader uses): register the file, PUT its content with curl,
    commit it, then compare the MD5 Zenodo reports with the local file. A failed or uncommitted file is deleted from
    the draft and uploaded again. With `parallel` > 1 several files are uploaded at the same time (one thread and one
    HTTP session per file). Files that still fail after `tries` attempts are listed at the end; rerun to resume."""
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor
    files_url = f"{api}/records/{rec_id}/draft/files"
    tok = s.headers["Authorization"]
    lock = threading.Lock()

    def say(msg):
        with lock:
            print(f"  {time.strftime('%H:%M:%S')} {msg}", flush=True)

    def entries(sess):
        r = sess.get(files_url, timeout=60)
        r.raise_for_status()
        return {e["key"]: e for e in r.json().get("entries", [])}

    existing = entries(s)

    def one(item):
        path, name = item
        sess = requests.Session()
        sess.headers["Authorization"] = tok
        local_md5 = md5_of(path)
        current = existing.get(name)
        if current and current.get("status") == "completed" and current.get("checksum", "").replace("md5:", "") == local_md5:
            say(f"{name}: already in the draft, MD5 matches")
            return name, True
        for attempt in range(1, tries + 1):
            t_try = time.time()
            try:
                if attempt > 1 or name in existing:
                    r = sess.delete(f"{files_url}/{name}", timeout=60)
                    if r.status_code not in (204, 404):
                        r.raise_for_status()
                r = sess.post(files_url, json=[{"key": name}], timeout=60)
                r.raise_for_status()
                say(f"{name}: uploading {os.path.getsize(path) / 1e9:.2f} GB (attempt {attempt})")
                # The token goes to curl through a config on stdin (-K -), never on the command line (visible in `ps`).
                cp = subprocess.run(["curl", "-K", "-", "--fail-with-body", "-sS", "-X", "PUT",
                                     "-H", "Content-Type: application/octet-stream", "--upload-file", path,
                                     "--retry", "0", "--speed-limit", "10000", "--speed-time", "600",
                                     f"{files_url}/{name}/content"], input=f'header = "Authorization: {tok}"\n',
                                    capture_output=True, text=True)
                if cp.returncode != 0:
                    raise RuntimeError(f"curl exit {cp.returncode}: {cp.stderr.strip()[:200]}")
                r = sess.post(f"{files_url}/{name}/commit", timeout=600)
                r.raise_for_status()
                remote = r.json().get("checksum", "").replace("md5:", "")
                if remote != local_md5:
                    raise ValueError(f"MD5 mismatch: local {local_md5}, Zenodo {remote}")
                dt = time.time() - t_try
                say(f"{name}: uploaded, MD5 verified ({dt / 60:.1f} min, {os.path.getsize(path) / max(dt, 1) / 1e6:.2f} MB/s)")
                return name, True
            except (requests.RequestException, ValueError, RuntimeError, OSError) as e:
                say(f"{name}: attempt {attempt} failed after {(time.time() - t_try) / 60:.1f} min: {str(e)[:200]}")
                if attempt < tries:
                    # 429 (rate limit) and gateway errors: back off longer before the next attempt
                    time.sleep((300 if "429" in str(e) else 60) * attempt)
        return name, False

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        results = list(pool.map(one, list(zip(paths, names))))
    failed = [n for n, ok in results if not ok]
    if failed:
        sys.exit(f"{len(failed)} file(s) not uploaded: {', '.join(failed)}; rerun the same command to resume")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--record", type=int, required=True, help="record index in the manifest (1, 2, ...)")
    ap.add_argument("--metadata", default=None, help="JSON file with the Zenodo 'metadata' object; omit to upload files only and keep the draft's metadata")
    ap.add_argument("--extra", nargs="*", default=[], help="additional files to upload (README.md, LICENSE.txt)")
    ap.add_argument("--deposition", type=int, default=None, help="reuse an existing draft instead of creating one")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="use the old deposit/bucket API (failed with 502 on 10 GB files)")
    ap.add_argument("--tries", type=int, default=5, help="attempts per file")
    ap.add_argument("--parallel", type=int, default=1, help="files uploaded at the same time (default 1)")
    ap.add_argument("--prune", action="store_true", help="delete draft files that are not part of this record's plan")
    ap.add_argument("--describe-dir", default=None,
                    help="directory from zenodo_describe.py; sets only title and description of the draft")
    a = ap.parse_args()

    api = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token()}"

    manifest = json.load(open(a.manifest))
    rec = next((r for r in manifest["records"] if r["index"] == a.record), None)
    if rec is None:
        sys.exit(f"record {a.record} not in manifest")
    rec_dir = os.path.join(os.path.dirname(os.path.abspath(a.manifest)), "records", f"record_{a.record:02d}")
    paths = [os.path.realpath(os.path.join(rec_dir, p)) for p in rec["pieces"]]
    paths.append(os.path.join(rec_dir, "SHA256SUMS.txt"))
    paths += [os.path.realpath(p) for p in a.extra]
    # Names come from the plan, not from the resolved paths: a piece may be a symlink to a file with another name.
    names = list(rec["pieces"]) + ["SHA256SUMS.txt"] + [os.path.basename(p) for p in a.extra]
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f"missing file: {p}")
    print(f"record {a.record}: {len(paths)} files, {sum(os.path.getsize(p) for p in paths) / 1e9:.2f} GB")

    if a.deposition:
        dep = s.get(f"{api}/deposit/depositions/{a.deposition}", timeout=60)
    else:
        dep = s.post(f"{api}/deposit/depositions", json={}, timeout=60)
    if not dep.ok and a.deposition and not a.legacy:
        # The legacy deposit endpoint is only needed for the bucket of --legacy uploads; the InvenioRDM path works
        # with the record id alone (the deposit endpoint occasionally answers 500).
        print(f"WARNING: legacy deposit endpoint answered {dep.status_code}; continuing with the draft-files API", flush=True)
        dep = {"id": a.deposition, "links": {"bucket": None, "html": f"https://zenodo.org/uploads/{a.deposition}"}, "metadata": {}}
    else:
        dep.raise_for_status()
        dep = dep.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    if a.metadata:
        metadata = json.load(open(a.metadata))
        metadata.setdefault("prereserve_doi", True)
        r = s.put(f"{api}/deposit/depositions/{dep_id}", json={"metadata": metadata}, timeout=60)
        if not r.ok:
            sys.exit(f"metadata rejected: {r.status_code} {r.text}")
        dep = r.json()
    doi = (dep.get("metadata", {}).get("prereserve_doi") or {}).get("doi")
    print(f"draft {dep_id}  reserved DOI {doi}  bucket {bucket}")

    if a.describe_dir:
        set_title_description(s, api, dep_id, a.describe_dir, a.record)
    if a.prune and not a.legacy:
        prune_rdm(s, api, dep_id, set(names))

    if a.legacy:
        upload_legacy(s, api, dep_id, bucket, paths, names)
    else:
        upload_rdm(s, api, dep_id, paths, names, a.tries, a.parallel)

    print(f"\nDone. Draft {dep_id} is ready for review; publish it in the web interface: {dep['links'].get('html')}")
    print(f"Record index {a.record} -> Zenodo id {dep_id} (for `zenodo_pack.py urls --record-ids {a.record}:{dep_id},...`)")


if __name__ == "__main__":
    main()
