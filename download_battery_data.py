# -*- coding: utf-8 -*-
"""Download the lithium-ion battery EIS spectrum used in the manuscript.

The battery data are taken from the public dataset:
    H. Mustafa et al., "SoC estimation on Li-ion batteries: A new EIS-based
    dataset for data-driven applications," Data in Brief 57, 110947 (2024).
    Dataset DOI: 10.17632/cb887gkmxw (Mendeley Data, Version 2)
    Article DOI: 10.1016/j.dib.2024.110947

The manuscript analysis (Fig. 7 and Section 4.2) uses one representative
spectrum at 5% state of charge, reported there as
"Hk_IFR14500_SoC_5_04-07-2023_05"; the full file name in the dataset is
"Hk_IFR14500_SoC_5_04-07-2023_05-13.csv" (LFP cell, 5% SoC, 0.01-1000 Hz,
four-terminal measurement).

This script queries the Mendeley Data public API, locates the file by name,
downloads it into the sibling "6 Case Data" folder, and verifies the SHA-256
checksum published by the repository. Only the Python standard library is
required.

Usage:
    python download_battery_data.py
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATASET_ID = "cb887gkmxw"
API_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
FILE_PREFIX = "Hk_IFR14500_SoC_5_04-07-2023_05"
OUTPUT_DIR = Path(__file__).resolve().parent / "6 Case Data"


def fetch(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def main():
    print(f"Querying Mendeley Data dataset {DATASET_ID} ...")
    with fetch(API_URL, timeout=60) as resp:
        dataset = json.load(resp)

    matches = [f for f in dataset.get("files", [])
               if f.get("filename", "").startswith(FILE_PREFIX)]
    if not matches:
        sys.exit(f"ERROR: no file starting with '{FILE_PREFIX}' found in the dataset.")
    entry = matches[0]
    name = entry["filename"]
    details = entry["content_details"]
    url = details["download_url"]
    expected_sha256 = details.get("sha256_hash")
    print(f"Found: {name} ({details.get('size', '?')} bytes)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / name
    print(f"Downloading to: {out_path}")
    with fetch(url, timeout=120) as resp:
        data = resp.read()
    out_path.write_bytes(data)

    if expected_sha256:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            out_path.unlink(missing_ok=True)
            sys.exit(f"ERROR: checksum mismatch (expected {expected_sha256}, got {actual}).")
        print(f"SHA-256 verified: {actual}")
    print("Done.")


if __name__ == "__main__":
    main()
