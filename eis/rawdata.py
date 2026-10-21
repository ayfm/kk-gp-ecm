"""Raw data of D1, D2 and D3: download, MD5 check, workbook build and comparison.

  D1  Zhang et al. (2020), Zenodo record 3633835, the file "EIS data.zip" (26957676 bytes)
  D2  Rashid et al. (2023), Mendeley Data mn9fb7xdx6 version 3, the file "DIB_Data.zip"
      (338496889 bytes)
  D3  Mustafa et al. (2024), Mendeley Data cb887gkmxw version 2, all files as one zip archive
      (about 0.8 MB)

For one dataset, `fetch` saves the archive in <raw>/archives/. An archive that is already there
is used again. A download that ends before the size given by the server raises an error and
saves no archive. The function checks the MD5 checksum of the D1 archive and the SHA-256 checksum
of the D2 archive given by the two repositories, extracts the archive and moves the folder that
the build modules read to <raw>/D1/EIS data/EIS data, <raw>/D2/DIB_Data or <raw>/D3. The MD5
checksum of every raw file that eis.build_d1, eis.build_d2 and eis.build_d3 read (513 files) is
checked by `check_files` against datasets/raw_files.md5. Last, `build_workbooks` builds the
three workbooks from <raw>, and `compare_workbooks` compares a rebuilt workbook with a workbook
of datasets/.
"""

import hashlib
import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from . import build_d1, build_d2, build_d3, env

SOURCES = {
    "D1": {
        "url": "https://zenodo.org/api/records/3633835/files/EIS%20data.zip/content",
        "archive": "D1_zenodo_3633835_EIS_data.zip",
        "size": 26957676,
        "hash": ("md5", "9f31069d770d7d323d8e2bf21267afe2"),
        # the folder that holds this file is moved to <raw>/D1/EIS data/EIS data
        "marker": "EIS_state_I_25C01.txt",
        "target": os.path.join("D1", "EIS data", "EIS data"),
    },
    "D2": {
        "url": ("https://data.mendeley.com/public-files/datasets/mn9fb7xdx6/files/"
                "1c1c1344-6f84-4b88-862b-2156098ccc3c/file_downloaded"),
        "archive": "D2_mendeley_mn9fb7xdx6_v3_DIB_Data.zip",
        "size": 338496889,
        "hash": ("sha256", "54bad35157924ec4e61cac91c98ac0501cf8949ea556c2c8fecf46c94540f09b"),
        # the folder that holds .csvfiles/EIS_Test is moved to <raw>/D2/DIB_Data
        "marker": os.path.join(".csvfiles", "EIS_Test"),
        "target": os.path.join("D2", "DIB_Data"),
    },
    "D3": {
        "url": "https://data.mendeley.com/public-api/zip/cb887gkmxw/download/2",
        "archive": "D3_mendeley_cb887gkmxw_v2.zip",
        "size": None,
        "hash": None,
        # the folder that holds B01/EIS measurements is moved to <raw>/D3
        "marker": os.path.join("B01", "EIS measurements"),
        "target": "D3",
    },
}


def file_hash(path: str, kind: str) -> str:
    h = hashlib.new(kind)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def archive_ok(path: str, src: dict) -> bool:
    if not os.path.isfile(path):
        return False
    if src["size"] is not None and os.path.getsize(path) != src["size"]:
        return False
    if src["hash"] is not None and file_hash(path, src["hash"][0]) != src["hash"][1]:
        return False
    return True


def download(url: str, path: str) -> None:
    part = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "kk-gp-ecm download_raw.py"})
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, step = 0, 0
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            done += len(block)
            if total and done * 10 // total > step:
                step = done * 10 // total
                print(f"    {done / 1e6:.0f} of {total / 1e6:.0f} MB", flush=True)
    # r.read returns an empty block without an error when the connection closes early
    if total and done != total:
        raise RuntimeError(f"download of {url} stopped after {done} of {total} bytes")
    os.replace(part, path)


def find_folder(top: str, marker: str) -> str:
    """Return the folder under `top` (without __MACOSX) that holds the file or folder `marker`."""
    for root, dirs, _files in os.walk(top):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        if os.path.exists(os.path.join(root, marker)):
            return root
    raise RuntimeError(f"{marker} not found in the extracted archive under {top}")


def fetch(name: str, raw: str) -> None:
    src = SOURCES[name]
    archives = os.path.join(raw, "archives")
    os.makedirs(archives, exist_ok=True)
    path = os.path.join(archives, src["archive"])
    if archive_ok(path, src):
        print(f"{name}: using {path}")
    else:
        print(f"{name}: downloading {src['url']}", flush=True)
        download(src["url"], path)
        if not archive_ok(path, src):
            raise RuntimeError(f"{name}: size or checksum of {path} differs from the repository")
    target = os.path.join(raw, src["target"])
    if os.path.exists(target):
        print(f"{name}: {target} exists, extraction skipped")
        return
    tmp = os.path.join(raw, f"extract_{name}")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    shutil.move(find_folder(tmp, src["marker"]), target)
    shutil.rmtree(tmp)
    print(f"{name}: extracted to {target}")


def check_files(raw: str, names: list) -> bool:
    """Check the MD5 checksum of every file of raw_files.md5 that belongs to the datasets `names`.

    Prints the result and returns True when every file matches.
    """
    ok, n = True, 0
    with open(os.path.join(env.DIR_DATASETS, "raw_files.md5")) as f:
        for line in f:
            digest, rel = line.rstrip("\n").split("  ", 1)
            if rel.split("/", 1)[0] not in names:
                continue
            path = os.path.join(raw, *rel.split("/"))
            n += 1
            if not os.path.isfile(path):
                print(f"missing: {rel}")
                ok = False
            elif file_hash(path, "md5") != digest:
                print(f"MD5 differs: {rel}")
                ok = False
    print(f"{n} raw files checked: {'all MD5 values equal raw_files.md5' if ok else 'differences found'}")
    return ok


def build_workbooks(raw: str, out_dir: str) -> list:
    """Build the three workbooks from the raw data in `raw` (the layout of `fetch`) into
    `out_dir`, and return their paths."""
    build_d1.RAW_ROOT = os.path.join(raw, "D1", "EIS data", "EIS data")
    build_d2.RAW_ROOT = os.path.join(raw, "D2", "DIB_Data", ".csvfiles", "EIS_Test")
    build_d3.RAW_ROOT = os.path.join(raw, "D3")
    return [build_d1.build(os.path.join(out_dir, "D1_zhang_dataset.xlsx")),
            build_d2.build(os.path.join(out_dir, "D2_rashid_dataset.xlsx")),
            build_d3.build(os.path.join(out_dir, "D3_mustafa_dataset.xlsx"))]


def compare_workbooks(rebuilt: str, reference: str, atol: float = 1e-12):
    """Compare a rebuilt workbook with a reference workbook: sheets, columns, rows and values.

    Both workbooks must have the same sheet names and, in each sheet, the same columns and the
    same number of rows. Returns (equal, table). The table has one row per column of every sheet
    with the largest absolute difference of a numeric column (two empty cells count as equal) or
    a note. `equal` is False when a difference exceeds `atol` or when a sheet name, a column
    name, an empty cell or a text cell differs.
    """
    a = pd.read_excel(rebuilt, sheet_name=None, engine="openpyxl")
    b = pd.read_excel(reference, sheet_name=None, engine="openpyxl")
    if list(a) != list(b):
        return False, pd.DataFrame([{"sheet": "", "column": "", "max_abs_diff": np.nan,
                                     "note": f"sheet names differ: {list(a)} {list(b)}"}])
    ok = True
    rows = []
    for sheet in a:
        x, y = a[sheet], b[sheet]
        if list(x.columns) != list(y.columns) or len(x) != len(y):
            rows.append({"sheet": sheet, "column": "", "max_abs_diff": np.nan,
                         "note": "columns or rows differ"})
            ok = False
            continue
        for col in x.columns:
            if pd.api.types.is_numeric_dtype(x[col]) and pd.api.types.is_numeric_dtype(y[col]):
                u, w = x[col].to_numpy(float), y[col].to_numpy(float)
                both_nan = np.isnan(u) & np.isnan(w)
                if (np.isnan(u) != np.isnan(w)).any():
                    rows.append({"sheet": sheet, "column": col, "max_abs_diff": np.nan,
                                 "note": "empty cells differ"})
                    ok = False
                    continue
                d = float(np.max(np.abs(u[~both_nan] - w[~both_nan]), initial=0.0))
                rows.append({"sheet": sheet, "column": col, "max_abs_diff": d, "note": ""})
                ok &= d <= atol
            elif not x[col].equals(y[col]):
                rows.append({"sheet": sheet, "column": col, "max_abs_diff": np.nan,
                             "note": "values differ"})
                ok = False
    return ok, pd.DataFrame(rows)
