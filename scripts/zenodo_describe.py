#!/usr/bin/env python3
"""
Write the Zenodo title and description of every record of a `zenodo_pack.py` package (profiles for CESNET-TimeSeries26 and CESNET-TimeSeries24).

For each record: the list of files with sizes (and, for split files, which parts are where), links to the other
records, the dataset description, usage (library and manual re-assembly with the expected SHA-256) and citation.
Output: <out>/record_<k>_title.txt, <out>/record_<k>_description.html (paste into the Zenodo description editor's
source view, or send through the API) and <out>/record_<k>_description.txt (plain text).

Example:
  python scripts/zenodo_describe.py --manifest .../pack2/manifest.json \\
      --record-ids 1:23278901,2:23280650,3:23280754,4:23283552 --out .../pack2/descriptions
"""

import argparse
import html
import json
import os
import re

PROFILES = {
    "CESNET-TimeSeries26": {
        "main_record": "https://zenodo.org/records/21945629",
        "main_doi": "10.5281/zenodo.21945629",
        "title": "CESNET-TimeSeries26: Multi-year network traffic time series from the CESNET3 ISP network (HDF5 files, part {k} of {n})",
        "class": "CESNET_TimeSeries26",
        "about": [
            "CESNET-TimeSeries26 is a multi-year collection of network traffic time series derived from long-term IP-flow "
            "monitoring of CESNET3, the Czech national research and education network. It is the successor of "
            "CESNET-TimeSeries24 and extends its 40-week collection to 996 days (9 October 2023 – 30 June 2026), which makes "
            "annual seasonality, year-over-year drift and model-longevity studies possible on real operational traffic.",
            "The dataset has ten source types, each in three aggregations (10 minutes, 1 hour, 1 day), 30 files and 178.7 GB "
            "in total: institutions (285 series); institution_subnets (566); ip_addresses_full (278,466, every monitored IP "
            "address); ip_addresses_sample (1,000, uniform random sample); five gap-stratified samples "
            "ip_addresses_sample_gaps_20_00 … _100_80 (1,000 series each, by the share of missing 10-minute windows); and "
            "subnet_longitudinal (135 IP addresses of one institution subnet, 31 December 2021 – 9 July 2026, TCP traffic only, "
            "own time grid).",
            "Each datapoint is multivariate: numbers of flows, packets and bytes; numbers of unique destination IP addresses, "
            "ASNs and transport-layer ports; TCP/UDP and direction ratios of packets and bytes; average flow duration and "
            "average TTL. Entity identifiers are pseudonymous integers; no IP addresses, institution names or other "
            "identifying information are published.",
        ],
        "citation": "Please cite the dataset record: CESNET-TimeSeries26. Zenodo. https://doi.org/10.5281/zenodo.21945629. The "
                    "dataset paper is in preparation; until it is published, please also cite: Koumar, J., Hynek, K., Čejka, T., "
                    "Šiška, P. CESNET-TimeSeries24: Time series dataset for network traffic anomaly detection and forecasting. "
                    "Scientific Data 12, 338 (2025). https://doi.org/10.1038/s41597-025-04603-x",
    },
    "CESNET-TimeSeries24": {
        "main_record": "https://zenodo.org/records/13382427",
        "main_doi": "10.5281/zenodo.13382427",
        "title": "CESNET-TimeSeries24: Time series dataset for network traffic anomaly detection and forecasting (HDF5 files for CESNET TS-Zoo, part {k} of {n})",
        "class": "CESNET_TimeSeries24",
        "about": [
            "CESNET-TimeSeries24 was collected by long-term monitoring of selected statistical metrics for 40 weeks for each "
            "IP address on CESNET3, the Czech national research and education network. It covers network traffic of more "
            "than 275,000 active IP addresses (office computers, NATs, servers, Wi-Fi routers, honeypots, game consoles in "
            "dormitories, ...) and contains real network anomalies of all kinds. It was created from 66 billion IP flows "
            "carrying 4 trillion packets and about 3.7 PB of data.",
            "The time series are provided on four levels (source types) - institutions, institution_subnets, "
            "ip_addresses_full (every monitored IP address) and ip_addresses_sample (uniform random sample) - each in three "
            "aggregations (10 minutes, 1 hour, 1 day): 12 HDF5 files.",
            "Each datapoint is multivariate: numbers of flows, packets and bytes; numbers of unique destination IP addresses, "
            "ASNs and transport-layer ports; TCP/UDP and direction ratios of packets and bytes; average flow duration and "
            "average TTL. Entity identifiers are pseudonymous integers.",
            "This distribution holds the HDF5 (PyTables) files of the CESNET TS-Zoo library, converted from the original CSV "
            "release without altering the measured values; coarser aggregations are computed from the 10-minute data. The "
            "original release, documentation and licence are in the main record. Licence: CC BY 4.0, as the original.",
        ],
        "citation": "Please cite: Koumar, J., Hynek, K., Čejka, T., Šiška, P. CESNET-TimeSeries24: Time series dataset for "
                    "network traffic anomaly detection and forecasting. Scientific Data 12, 338 (2025). "
                    "https://doi.org/10.1038/s41597-025-04603-x, and the dataset: Zenodo, https://doi.org/10.5281/zenodo.13382427. "
                    "If you use the library, please also cite: Kureš, M., Koumar, J., Hynek, K. CESNET TS-Zoo: A Library for "
                    "Reproducible Analysis of Network Traffic Time Series. CNSM 2025.",
    },
}


def size(b: int) -> str:
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"


def ranges(nums: list[int]) -> str:
    nums = sorted(nums)
    out, start, prev = [], nums[0], nums[0]
    for x in nums[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        out.append(f"{start}" if start == prev else f"{start}–{prev}")
        if x is not None:
            start = prev = x
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--record-ids", required=True, help="1:23278901,2:23280650,...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = json.load(open(a.manifest))
    P = PROFILES[m["database"]]
    MAIN_RECORD, MAIN_DOI, TITLE, ABOUT, CITATION, CLS = P["main_record"], P["main_doi"], P["title"], P["about"], P["citation"], P["class"]
    gz = bool(m.get("gzip"))
    ids = {int(k): v for k, v in (x.split(":") for x in a.record_ids.split(","))}
    n = len(m["records"])
    rec_of = {p: r["index"] for r in m["records"] for p in r["pieces"]}
    piece_info = {}
    for f in m["files"]:
        k = len(f["pieces"])
        for i, p in enumerate(f["pieces"], start=1):
            piece_info[p["name"]] = (f, i, k, p["bytes"])
    os.makedirs(a.out, exist_ok=True)

    for r in m["records"]:
        k = r["index"]
        whole, split = [], {}
        for p in r["pieces"]:
            f, i, cnt, b = piece_info[p]
            if cnt == 1:
                whole.append((p, b))
            else:
                split.setdefault(f["name"], []).append((i, b))
        others = [(j, ids[j]) for j in sorted(ids) if j != k]

        lines = [f"Part {k} of {n} of the HDF5 distribution of {m['database']} for the CESNET TS-Zoo library "
                 f"(https://github.com/CESNET/cesnet-tszoo). The main record of the dataset is {MAIN_RECORD} "
                 f"(DOI: {MAIN_DOI}).", "", f"FILES IN THIS RECORD ({size(r['bytes'])}, {len(r['pieces'])} files + SHA256SUMS.txt)"]
        h = [f"<p>{html.escape(lines[0])}</p>", f"<p><strong>{html.escape(lines[2])}</strong></p><ul>"]
        for name, b in sorted(whole):
            lines.append(f"- {name} ({size(b)})")
            h.append(f"<li>{html.escape(name)} ({size(b)})</li>")
        for name, parts in sorted(split.items()):
            f = next(x for x in m["files"] if x["name"] == name)
            total = len(f["pieces"])
            nums = [i for i, _ in parts]
            txt = f"- {name}.NNN{'.gz' if gz else ''}: part{'s' if len(nums) > 1 else ''} {ranges(nums)} of {total} ({size(sum(b for _, b in parts))})"
            lines.append(txt)
            h.append(f"<li>{html.escape(txt[2:])}</li>")
        lines.append("- SHA256SUMS.txt (SHA-256 of every file above)")
        h.append("<li>SHA256SUMS.txt (SHA-256 of every file above)</li></ul>")

        lines += ["", "OTHER PARTS OF THIS DISTRIBUTION"]
        h.append("<p><strong>Other parts of this distribution</strong></p><ul>")
        for j, rid in others:
            lines.append(f"- Part {j}: https://zenodo.org/records/{rid}")
            h.append(f'<li>Part {j}: <a href="https://zenodo.org/records/{rid}">https://zenodo.org/records/{rid}</a></li>')
        h.append("</ul>")

        lines += ["", "ABOUT THE DATASET"] + ABOUT
        h.append("<p><strong>About the dataset</strong></p>" + "".join(f"<p>{html.escape(t)}</p>" for t in ABOUT))

        lines += ["", "USAGE", "With the library, files and their parts are downloaded from all records, re-assembled and "
                  "verified automatically:",
                  f"    from cesnet_tszoo.datasets import {CLS}",
                  f'    ds = {CLS}.get_dataset(data_root="...", source_type="institutions", aggregation="1_hour", dataset_type="time_based")']
        h.append("<p><strong>Usage</strong></p><p>With the library, files and their parts are downloaded from all records, "
                 f"re-assembled and verified automatically:</p><pre>from cesnet_tszoo.datasets import {CLS}\n"
                 f"ds = {CLS}.get_dataset(data_root=\"...\", source_type=\"institutions\", "
                 "aggregation=\"1_hour\", dataset_type=\"time_based\")</pre>")
        if gz:
            note = ("Every file is stored gzip-compressed (.gz); a split file is a sequence of independent gzip members, so "
                    "concatenating its parts and decompressing gives the original HDF5 file. Single-piece files: gunzip <file>.h5.gz")
            lines.append(note)
            h.append(f"<p>{html.escape(note)}</p>")
        if split:
            lines.append("Without the library, download all parts of a split file (they may be spread over several parts of "
                         "this distribution), concatenate them in numeric order and verify the result:")
            h.append("<p>Without the library, download all parts of a split file (they may be spread over several parts "
                     "of this distribution), concatenate them in numeric order and verify the result:</p>")
            for name in sorted(split):
                f = next(x for x in m["files"] if x["name"] == name)
                where = {}
                for i, p in enumerate(f["pieces"], start=1):
                    where.setdefault(rec_of[p["name"]], []).append(i)
                loc = "; ".join(f"part{'s' if len(v) > 1 else ''} {ranges(v)} in part {rk}" for rk, v in sorted(where.items()))
                cmd = (f"cat {name}.*.gz | gunzip > {name}   # {loc}" if gz else f"cat {name}.* > {name}   # {loc}") + f"\n    expected SHA-256: {f['sha256']}"
                lines.append("    " + cmd)
                h.append(f"<pre>{html.escape(cmd)}</pre>")
        lines.append("The downloaded files of this record can be verified with: shasum -a 256 -c SHA256SUMS.txt")
        h.append("<p>The downloaded files of this record can be verified with <code>shasum -a 256 -c SHA256SUMS.txt</code>.</p>")

        lines += ["", "CITATION", CITATION, "", "CONTACT", "Josef Koumar, CESNET / Czech Technical University in Prague"]
        h.append(f"<p><strong>Citation</strong></p><p>{html.escape(CITATION)}</p>"
                 "<p><strong>Contact</strong>: Josef Koumar, CESNET / Czech Technical University in Prague.</p>")
        h = [re.sub(r"(https://[^\s<,;)]+?)(?=[\s<,;)]|\.?$)", r'\1', x) for x in h]

        open(os.path.join(a.out, f"record_{k}_title.txt"), "w").write(TITLE.format(k=k, n=n) + "\n")
        open(os.path.join(a.out, f"record_{k}_description.txt"), "w").write("\n".join(lines) + "\n")
        open(os.path.join(a.out, f"record_{k}_description.html"), "w").write("".join(h) + "\n")
        print(f"record {k}: {len(r['pieces'])} files, {size(r['bytes'])}")


if __name__ == "__main__":
    main()
