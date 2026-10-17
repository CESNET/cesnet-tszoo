#!/usr/bin/env python3
"""Fill BACKUP_DRIVE_IDS of CESNET-TimeSeries26 from an rclone listing of the backup Google Drive folder.

Usage (from the repository root):
    rclone lsjson "gdrive:TSZoo Datasets/CESNET-TimeSeries26" > ts26_backup.json
    python scripts/ts26_backup_drive_ids.py ts26_backup.json

Only files whose name matches a dataset of the database are written; the block between `BACKUP_DRIVE_IDS` and the
next blank line in constants.py is replaced. The files must also be shared as "Anyone with the link" for the
direct-download URL to work.
"""
import json
import re
import sys
from pathlib import Path

CONSTANTS = Path(__file__).resolve().parents[1] / "cesnet_tszoo/datasets/databases/cesnet_timeseries26/constants.py"


def main(listing: str) -> None:
    items = json.load(open(listing))
    text = CONSTANTS.read_text(encoding="utf-8")
    known = set(re.findall(r'"(CESNET-TimeSeries26-[^"]+)": \[google_drive_url', text))
    ids = {}
    for it in items:
        name = it.get("Name", "")
        if name.endswith(".h5") and name[:-3] in known and it.get("ID"):
            ids[name[:-3]] = it["ID"]
    missing = sorted(known - set(ids))
    body = "BACKUP_DRIVE_IDS: dict[str, str] = {\n" + "".join(f'    "{k}": "{v}",\n' for k, v in sorted(ids.items())) + "}\n"
    new, n = re.subn(r"BACKUP_DRIVE_IDS: dict\[str, str\] = \{.*?\}\n", body, text, count=1, flags=re.S)
    if n != 1:
        sys.exit("BACKUP_DRIVE_IDS block not found in constants.py")
    CONSTANTS.write_text(new, encoding="utf-8")
    print(f"{len(ids)} IDs written; missing on the backup Drive: {len(missing)}" + (f" ({', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main(sys.argv[1])
