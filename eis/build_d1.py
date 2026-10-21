"""Build D1_zhang_dataset.xlsx from the raw files of Zhang et al. (2020), Zenodo record 3633835.

The raw files are <raw>/D1/EIS data/EIS data/EIS_state_<state>_<T>C<cell>.txt from the archive
"EIS data.zip" of the record (`eis.rawdata.fetch`). There is one file per state, temperature T in
degrees Celsius and cell, and a file holds the spectra of several cycles (column "cycle number").

The build keeps T = 25, the cells 1 to 8, the cycle numbers up to 5 and the states I, II, III, IV,
V and IX. The spectrum of a state is the mean over the kept files and cycles at each frequency.
The raw files store -Im Z, and the workbook stores Im Z. The raw files have no voltage column, so
each state gets the nominal label v of datasets/D1_state_ocv_map.json. This label is also the
name of its sheet.

The workbook has one sheet per state. Its columns f (Hz), zr, zj, zmod (ohm) and zphz (degrees)
hold the means of the raw columns, idc is empty and vdc holds the label v. The sheet meta holds
v_nom and v_actual (both the label), n_points, f_min, f_max and the state name. Before it calls
`build`, eis.rawdata.build_workbooks sets RAW_ROOT.
"""

import json
import os
from glob import glob

import numpy as np
import pandas as pd

from . import env

logger = env.logger

DATASET_NAME = "D1_zhang"
# folder of the raw files (set by eis.rawdata.build_workbooks)
RAW_ROOT = os.path.join(env.DIR_DATASETS, "raw", "D1", "EIS data", "EIS data")
OCV_MAP_PATH = os.path.join(env.DIR_DATASETS, "D1_state_ocv_map.json")
OUTPUT_PATH = os.path.join(env.DIR_DATASETS, "rebuilt", f"{DATASET_NAME}_dataset.xlsx")

# kept files and states
TEMP_KEEP = 25
CELLS_KEEP = {1, 2, 3, 4, 5, 6, 7, 8}
MAX_AGING_CYCLE = 5
STATES = ["I", "II", "III", "IV", "V", "IX"]


def load_ocv_map() -> dict[str, float]:
    with open(OCV_MAP_PATH) as f:
        raw = json.load(f)
    return {s: float(raw[s]) for s in STATES}


def parse_filename(path: str) -> tuple[str, int, int] | None:
    """Returns (state, temp_C, cell_id), or None if the file name does not match."""
    name = os.path.basename(path).removesuffix(".txt")
    # for example EIS_state_I_25C01
    if not name.startswith("EIS_state_"):
        return None
    tail = name.removeprefix("EIS_state_")
    parts = tail.rsplit("_", 1)
    if len(parts) != 2:
        return None
    state = parts[0]
    temp_cell = parts[1]  # for example "25C01"
    if "C" not in temp_cell:
        return None
    temp_str, cell_str = temp_cell.split("C")
    try:
        return state, int(temp_str), int(cell_str)
    except ValueError:
        return None


def read_eis_file(path: str) -> pd.DataFrame:
    """Read one raw EIS file. Returns one row per (cycle, frequency) point."""
    df = pd.read_csv(path, sep="\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "time/s": "t",
        "cycle number": "cycle",
        "freq/Hz": "f",
        "Re(Z)/Ohm": "zr",
        "-Im(Z)/Ohm": "neg_zj",
        "|Z|/Ohm": "zmod",
        "Phase(Z)/deg": "zphz",
    })
    df["zj"] = -df["neg_zj"]  # the raw column is -Im Z
    return df[["t", "cycle", "f", "zr", "zj", "zmod", "zphz"]]


def build_subset_for_state(state: str, ocv: float) -> pd.DataFrame:
    """Mean spectrum of one state over the kept cells and cycles."""
    pattern = os.path.join(RAW_ROOT, f"EIS_state_{state}_*.txt")
    files = sorted(glob(pattern))
    rows: list[pd.DataFrame] = []
    for path in files:
        meta = parse_filename(path)
        if meta is None:
            continue
        st, temp, cell = meta
        if temp != TEMP_KEEP or cell not in CELLS_KEEP:
            continue
        df = read_eis_file(path)
        df = df[df["cycle"] <= MAX_AGING_CYCLE].copy()
        if df.empty:
            continue
        df["cell"] = cell
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No data found for state {state}")
    out = pd.concat(rows, ignore_index=True)

    # mean over cells and cycles at each frequency
    agg = (
        out.groupby("f", sort=False)[["zr", "zj", "zmod", "zphz"]]
        .mean()
        .reset_index()
    )

    # idc is not in the raw files (NaN), and vdc holds the nominal label of the state
    agg["idc"] = np.nan
    agg["vdc"] = ocv
    agg = agg[["f", "zr", "zj", "zmod", "zphz", "idc", "vdc"]]
    agg = agg.sort_values("f", ascending=False).reset_index(drop=True)
    return agg


def build(output_path: str = OUTPUT_PATH) -> str:
    ocv_map = load_ocv_map()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    meta_rows = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for state in STATES:
            ocv = ocv_map[state]
            spectrum = build_subset_for_state(state, ocv)
            sheet = str(ocv)
            spectrum.to_excel(writer, sheet_name=sheet, index=False)
            meta_rows.append({
                "v_nom": ocv,
                "v_actual": ocv,  # no measured voltage in the raw files
                "n_points": int(len(spectrum)),
                "f_min": float(spectrum["f"].min()),
                "f_max": float(spectrum["f"].max()),
                "state": state,
            })
            logger.info(
                "  state %4s  ->  OCV=%.3f V, %2d pts, f in [%.3f, %.1f] Hz",
                state, ocv, len(spectrum), spectrum["f"].min(), spectrum["f"].max(),
            )

        meta = pd.DataFrame(meta_rows)
        meta.to_excel(writer, sheet_name="meta", index=False)

    logger.info("Wrote %s (%d sheets)", output_path, len(STATES) + 1)
    return output_path
