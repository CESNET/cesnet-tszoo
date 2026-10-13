#!/usr/bin/env python3
"""
Prepare the HDF5 files of one cesnet-tszoo database for upload to Zenodo.

Zenodo accepts at most 50 GB per record, so large files are split into consecutive parts
(`<file>.h5.001`, `<file>.h5.002`, ...) that the library re-assembles after download
(`cesnet_tszoo.utils.download.parts_source`). Files stored without internal HDF5 compression
can additionally be gzipped part by part (`--gzip`): every part is an independent gzip member,
so the library decompresses them while concatenating. Files that already use an HDF5
compression filter (e.g. CESNET-TimeSeries26, blosc2:zstd) gain nothing from gzip - keep them raw.

Two modes:

  pack     split/compress the files, compute SHA-256 of the originals and of every uploaded piece,
           plan the distribution of the pieces over records (first-fit decreasing, `--record-limit`)
           and write `<out>/manifest.json` + `<out>/records/record_<n>/` directories with symlinks
           to the pieces of each record (upload one directory = one Zenodo record).

  urls     after the upload, turn the manifest + the Zenodo record ids into the `DOWNLOAD_URLS`
           entries (`ZENODO_SOURCES`) for `constants.py` (printed as Python) - `--record-ids 1:1234567,2:1234568,...`.

Example:
  python scripts/zenodo_pack.py pack --database CESNET-TimeSeries26 \\
      --files <h5_dir>/*.h5 --out <out>
  python scripts/zenodo_pack.py urls --manifest <out>/manifest.json --record-ids 1:1234567,2:1234568
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

GB = 10**9
CHUNK = 64 * 1024**2


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HashingWriter:
    """File-like wrapper that updates a SHA-256 digest with everything written through it."""

    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return self.raw.write(data)

    def flush(self):
        self.raw.flush()


def write_piece(src, n_bytes: int, dst_path: str, use_gzip: bool, whole_digest) -> tuple[int, str]:
    """Copies `n_bytes` from the open file `src` into `dst_path` (gzipped when requested) in one pass; updates `whole_digest` with the raw bytes and returns (size, sha256) of the written piece."""
    remaining = n_bytes
    with open(dst_path, "wb") as raw:
        writer = HashingWriter(raw)
        out = gzip.GzipFile(fileobj=writer, mode="wb", compresslevel=6, mtime=0) if use_gzip else writer
        try:
            while remaining > 0:
                chunk = src.read(min(CHUNK, remaining))
                if not chunk:
                    raise EOFError(f"unexpected end of file while writing {dst_path}")
                whole_digest.update(chunk)
                out.write(chunk)
                remaining -= len(chunk)
        finally:
            if use_gzip:
                out.close()
    return os.path.getsize(dst_path), writer.digest.hexdigest()


def write_part(path: str, offset: int, n_bytes: int, dst_path: str, use_gzip: bool) -> tuple[int, str]:
    """Writes bytes [offset, offset + n_bytes) of `path` into `dst_path` (gzipped when requested); returns (size, sha256) of the piece. Runs in a worker process."""
    with open(path, "rb") as src:
        src.seek(offset)
        return write_piece(src, n_bytes, dst_path, use_gzip, hashlib.sha256())


def pack_file(path: str, out_dir: str, part_size: int, use_gzip: bool, log, jobs: int = 1) -> dict:
    name = os.path.basename(path)
    size = os.path.getsize(path)
    t0 = time.time()
    entry = {"name": name, "bytes": size, "sha256": None, "gzip": use_gzip, "pieces": []}

    if size <= part_size and not use_gzip:
        # Upload the original file as it is.
        digest = sha256_of_file(path)
        entry["sha256"] = digest
        entry["pieces"].append({"name": name, "path": os.path.abspath(path), "bytes": size, "sha256": digest})
        log(f"{name}: single piece {size / GB:.2f} GB ({time.time() - t0:.0f}s)")
        return entry

    n_parts = max(1, -(-size // part_size))
    names = []
    for i in range(1, n_parts + 1):
        piece_name = f"{name}.{i:03d}" if n_parts > 1 else name
        names.append(piece_name + (".gz" if use_gzip else ""))

    if jobs > 1 and n_parts > 1:
        # Parts are independent (every gzip part is its own gzip member), so they are written by `jobs` processes;
        # the SHA-256 of the whole original file is computed alongside in a thread.
        with ThreadPoolExecutor(1) as th, ProcessPoolExecutor(jobs) as pool:
            whole = th.submit(sha256_of_file, path)
            futures = [pool.submit(write_part, path, (i - 1) * part_size, min(part_size, size - (i - 1) * part_size),
                                   os.path.join(out_dir, names[i - 1]), use_gzip) for i in range(1, n_parts + 1)]
            for i, fut in enumerate(futures, start=1):
                piece_size, piece_digest = fut.result()
                piece_path = os.path.join(out_dir, names[i - 1])
                entry["pieces"].append({"name": names[i - 1], "path": os.path.abspath(piece_path), "bytes": piece_size, "sha256": piece_digest})
                log(f"{name}: piece {i}/{n_parts} -> {names[i - 1]} {piece_size / GB:.2f} GB ({time.time() - t0:.0f}s)")
            entry["sha256"] = whole.result()
        return entry

    digest_whole = hashlib.sha256()
    with open(path, "rb") as src:
        for i in range(1, n_parts + 1):
            n_bytes = min(part_size, size - (i - 1) * part_size)
            piece_path = os.path.join(out_dir, names[i - 1])
            piece_size, piece_digest = write_piece(src, n_bytes, piece_path, use_gzip, digest_whole)
            entry["pieces"].append({"name": names[i - 1], "path": os.path.abspath(piece_path), "bytes": piece_size, "sha256": piece_digest})
            log(f"{name}: piece {i}/{n_parts} -> {names[i - 1]} {piece_size / GB:.2f} GB ({time.time() - t0:.0f}s)")
    entry["sha256"] = digest_whole.hexdigest()
    return entry


def plan_records_sequential(files: list[dict], record_limit: int, max_files: int = 99) -> list[dict]:
    """Fill records one after another, taking the files from the largest to the smallest and the pieces of each file in
    order, so the parts of a split file stay contiguous and span at most two neighbouring records whenever possible;
    the small files fill the remaining space of the last records. A record also holds at most `max_files` pieces
    (Zenodo allows 100 files per record; one slot is kept for SHA256SUMS.txt)."""
    records: list[dict] = []
    current = {"index": 1, "bytes": 0, "pieces": []}
    for f in sorted(files, key=lambda f: (-f["bytes"], f["name"])):
        for p in f["pieces"]:
            if current["bytes"] + p["bytes"] > record_limit or len(current["pieces"]) >= max_files:
                records.append(current)
                current = {"index": len(records) + 1, "bytes": 0, "pieces": []}
            current["pieces"].append(p["name"])
            current["bytes"] += p["bytes"]
    if current["pieces"]:
        records.append(current)
    return records


def plan_records(files: list[dict], record_limit: int, sequential: bool = False) -> list[dict]:
    """First-fit decreasing bin packing of all pieces into records of at most `record_limit` bytes (or the sequential plan, see `plan_records_sequential`)."""
    if sequential:
        return plan_records_sequential(files, record_limit)
    pieces = [(p["bytes"], f["name"], p["name"]) for f in files for p in f["pieces"]]
    too_big = [p for p in pieces if p[0] > record_limit]
    if too_big:
        raise SystemExit(f"pieces larger than the record limit: {too_big}; lower --part-size")
    records: list[dict] = []
    for size, file_name, piece_name in sorted(pieces, key=lambda p: (-p[0], p[1], p[2])):
        for rec in records:
            if rec["bytes"] + size <= record_limit:
                rec["pieces"].append(piece_name)
                rec["bytes"] += size
                break
        else:
            records.append({"index": len(records) + 1, "bytes": size, "pieces": [piece_name]})
    for rec in records:
        rec["pieces"].sort()
    return records


def cmd_pack(args):
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "pack.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    part_size = int(args.part_size * GB)
    record_limit = int(args.record_limit * GB)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = {"database": args.database, "part_size_bytes": part_size, "record_limit_bytes": record_limit, "gzip": args.gzip, "files": []}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
    done = {f["name"] for f in manifest["files"]}

    for path in sorted(args.files, key=os.path.getsize):
        if os.path.basename(path) in done:
            log(f"{os.path.basename(path)}: already in manifest, skipped")
            continue
        manifest["files"].append(pack_file(path, args.out, part_size, args.gzip, log, args.jobs))
        manifest["records"] = plan_records(manifest["files"], record_limit, args.sequential)
        json.dump(manifest, open(manifest_path, "w"), indent=1)

    manifest["records"] = plan_records(manifest["files"], record_limit, args.sequential)
    json.dump(manifest, open(manifest_path, "w"), indent=1)

    # One directory per record with symlinks to its pieces - upload a directory as one Zenodo record.
    piece_paths = {p["name"]: p["path"] for f in manifest["files"] for p in f["pieces"]}
    for rec in manifest["records"]:
        rec_dir = os.path.join(args.out, "records", f"record_{rec['index']:02d}")
        os.makedirs(rec_dir, exist_ok=True)
        for old in os.listdir(rec_dir):
            os.remove(os.path.join(rec_dir, old))
        for piece in rec["pieces"]:
            os.symlink(piece_paths[piece], os.path.join(rec_dir, piece))
        with open(os.path.join(rec_dir, "SHA256SUMS.txt"), "w") as f:
            for piece in rec["pieces"]:
                digest = next(p["sha256"] for fl in manifest["files"] for p in fl["pieces"] if p["name"] == piece)
                f.write(f"{digest}  {piece}\n")
        log(f"record {rec['index']}: {len(rec['pieces'])} pieces, {rec['bytes'] / GB:.2f} GB")

    total = sum(f["bytes"] for f in manifest["files"])
    uploaded = sum(p["bytes"] for f in manifest["files"] for p in f["pieces"])
    log(f"{len(manifest['files'])} files, {total / GB:.2f} GB original, {uploaded / GB:.2f} GB to upload, {len(manifest['records'])} records; manifest {manifest_path}")


def cmd_urls(args):
    manifest = json.load(open(args.manifest))
    record_ids = dict(item.split(":") for item in args.record_ids.split(","))
    record_of_piece = {piece: rec["index"] for rec in manifest["records"] for piece in rec["pieces"]}
    missing = {str(rec["index"]) for rec in manifest["records"]} - set(record_ids)
    if missing:
        raise SystemExit(f"missing Zenodo record ids for records {sorted(missing)}")

    lines = ["# Zenodo sources generated by scripts/zenodo_pack.py urls", f"{args.var}: dict[str, list] = {{"]
    checksums = ["CHECKSUMS = {"]
    for f in sorted(manifest["files"], key=lambda f: f["name"]):
        dataset_name = f["name"][:-3] if f["name"].endswith(".h5") else f["name"]
        urls = [f'zenodo_file_url({record_ids[str(record_of_piece[p["name"]])]}, "{p["name"]}")' for p in f["pieces"]]
        if len(f["pieces"]) == 1 and not f["gzip"]:
            lines.append(f'    "{dataset_name}": [{urls[0]}],')
        else:
            lines.append(f'    "{dataset_name}": [parts_source([')
            lines.extend(f"        {u}," for u in urls)
            lines.append(f'    ], gzip_parts={f["gzip"]})],')
        checksums.append(f'    "{dataset_name}": "{f["sha256"]}",')
    lines.append("}")
    checksums.append("}")
    print("\n".join(lines))
    print()
    print("\n".join(checksums))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--database", required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part-size", type=float, default=10.0, help="GB (10^9 bytes) per part, default 10")
    p.add_argument("--record-limit", type=float, default=48.0, help="GB per Zenodo record, default 48 (Zenodo allows 50)")
    p.add_argument("--gzip", action="store_true", help="gzip every part (only for files without HDF5 compression)")
    p.add_argument("--jobs", type=int, default=1, help="processes writing the parts of a split file in parallel (useful with --gzip)")
    p.add_argument("--sequential", action="store_true", help="keep the parts of a split file contiguous across records (instead of first-fit decreasing)")
    p.set_defaults(func=cmd_pack)
    u = sub.add_parser("urls")
    u.add_argument("--manifest", required=True)
    u.add_argument("--record-ids", required=True, help="record index -> Zenodo record id, e.g. 1:1234567,2:1234568")
    u.add_argument("--var", default="ZENODO_SOURCES", help="name of the generated dict (default ZENODO_SOURCES)")
    u.set_defaults(func=cmd_urls)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
