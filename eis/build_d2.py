"""Build D2_rashid_dataset.xlsx from the raw files of Rashid et al. (2023), Mendeley Data
mn9fb7xdx6 version 3.

The raw files are
<raw>/D2/DIB_Data/.csvfiles/EIS_Test/Cell<NN>_<SOH>SOH_<T>degC_<SOC>SOC_<cycles>.xls from the
archive DIB_Data.zip of the dataset (`eis.rawdata.fetch`). Each file is an Excel file of the old
binary format without a header row. Its columns are the frequency (Hz), Re Z and Im Z (ohm).
Reading these files needs the package xlrd.

The build keeps the files at 100% state of health and 25 degrees Celsius and at the states of
charge 5, 20, 50, 70 and 95%, of every cell that has such files (cells 28 to 32). The spectrum
of a state of charge is the mean over these cells at each frequency. Each state gets the label v
of datasets/D2_soc_ocv_map.json. This label is also the name of its sheet.

The workbook has one sheet per state with the columns f, zr and zj (the means), zmod and zphz
(modulus and phase in degrees of the mean), idc (empty) and vdc (the label v). The sheet meta
holds v_nom and v_actual (both the label), n_points, f_min, f_max, soc_pct and n_cells_averaged.
Before it calls `build`, eis.rawdata.build_workbooks sets RAW_ROOT.
"""

import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd

from . import env

logger = env.logger

DATASET_NAME = "D2_rashid"
# folder of the raw files (set by eis.rawdata.build_workbooks)
RAW_ROOT = os.path.join(env.DIR_DATASETS, "raw", "D2", "DIB_Data", ".csvfiles", "EIS_Test")
OCV_MAP_PATH = os.path.join(env.DIR_DATASETS, "D2_soc_ocv_map.json")
OUTPUT_PATH = os.path.join(env.DIR_DATASETS, "rebuilt", f"{DATASET_NAME}_dataset.xlsx")

SOH_KEEP = 100
TEMP_KEEP = 25
SOCS = [5, 20, 50, 70, 95]

FNAME_RE = re.compile(
    r"^Cell(?P<cell>\d+)_(?P<soh>\d+)SOH_(?P<temp>\d+)degC_(?P<soc>\d+)SOC_(?P<cycles>\d+)\.xls$"
)


def load_ocv_map() -> dict[int, float]:
    with open(OCV_MAP_PATH) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items() if not k.startswith("_")}


def parse_filename(path: str) -> dict | None:
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}


def read_eis_file(path: str) -> pd.DataFrame:
    """Read one raw .xls file: no header row, the columns f, zr, zj."""
    df = pd.read_excel(path, engine="xlrd", header=None,
                       names=["f", "zr", "zj"])
    df = df.dropna(how="any").reset_index(drop=True)
    return df


def build_subset_for_soc(soc: int, ocv: float) -> tuple[pd.DataFrame, int]:
    """Mean spectrum over the kept cells at (SOH_KEEP, TEMP_KEEP, soc), and the number of cells."""
    pattern = os.path.join(RAW_ROOT, f"Cell*_{SOH_KEEP}SOH_{TEMP_KEEP}degC_{soc:02d}SOC_*.xls")
    files = sorted(glob(pattern))
    if not files:
        # state of charge written without the leading zero
        pattern = os.path.join(RAW_ROOT, f"Cell*_{SOH_KEEP}SOH_{TEMP_KEEP}degC_*{soc}SOC_*.xls")
        files = [p for p in sorted(glob(pattern))
                 if parse_filename(p) is not None
                 and parse_filename(p)["soc"] == soc]
    if not files:
        raise RuntimeError(f"No files for SOC={soc} SOH={SOH_KEEP} T={TEMP_KEEP}")

    frames = []
    for path in files:
        df = read_eis_file(path)
        meta = parse_filename(path)
        df["cell"] = meta["cell"]
        frames.append(df)

    cat = pd.concat(frames, ignore_index=True)
    agg = cat.groupby("f", sort=False)[["zr", "zj"]].mean().reset_index()

    # modulus and phase of the mean
    z = agg["zr"].to_numpy() + 1j * agg["zj"].to_numpy()
    agg["zmod"] = np.abs(z)
    agg["zphz"] = np.degrees(np.angle(z))
    agg["idc"] = np.nan
    agg["vdc"] = ocv  # the label of the state (the raw files have no voltage column)
    agg = agg[["f", "zr", "zj", "zmod", "zphz", "idc", "vdc"]]
    agg = agg.sort_values("f", ascending=False).reset_index(drop=True)
    return agg, len(files)


def build(output_path: str = OUTPUT_PATH) -> str:
    ocv_map = load_ocv_map()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    meta_rows = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for soc in SOCS:
            ocv = ocv_map[soc]
            spectrum, n_cells = build_subset_for_soc(soc, ocv)
            sheet = str(ocv)
            spectrum.to_excel(writer, sheet_name=sheet, index=False)
            meta_rows.append({
                "v_nom": ocv,
                "v_actual": ocv,
                "n_points": int(len(spectrum)),
                "f_min": float(spectrum["f"].min()),
                "f_max": float(spectrum["f"].max()),
                "soc_pct": soc,
                "n_cells_averaged": n_cells,
            })
            logger.info(
                "  SOC %2d%%  ->  OCV=%.3f V, %2d pts, f in [%.3f, %.1f] Hz, n_cells=%d",
                soc, ocv, len(spectrum), spectrum["f"].min(), spectrum["f"].max(), n_cells,
            )

        meta = pd.DataFrame(meta_rows)
        meta.to_excel(writer, sheet_name="meta", index=False)

    logger.info("Wrote %s (%d sheets)", output_path, len(SOCS) + 1)
    return output_path
