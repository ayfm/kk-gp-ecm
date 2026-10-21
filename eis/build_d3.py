"""Build D3_mustafa_dataset.xlsx from the raw files of Mustafa et al. (2024), Mendeley Data
cb887gkmxw version 2.

The raw files are
<raw>/D3/B<NN>/EIS measurements/Test_<k>/Hioki/Hk_IFR14500_SoC_<SOC>_<date>_<time>.csv of the 11
cells B01 to B11 and the two test runs Test_1 and Test_2 (`eis.rawdata.fetch`). Their columns
are Frequency(Hz), R(ohm) = Re Z, X(ohm) = Im Z, V(V), T(deg C) and Range(Ohm).

The build keeps every cell, both test runs and the 20 states of charge 5, 10, ..., 100%. The
spectrum of a state of charge is the mean over its files at each frequency, also for the voltage
V. Its label v_actual is the mean over the files of the mean voltage of each file. The sheet name
v_nom is v_actual rounded to 4 decimals and raised by 0.001 V as long as it equals the name of an
earlier state.

The workbook has one sheet per state with the columns f, zr, zj and vdc (the means), zmod and
zphz (modulus and phase in degrees of the mean) and idc (empty). The sheet meta holds v_nom,
v_actual, n_points, f_min, f_max, soc_pct and n_files_averaged. Before it calls `build`,
eis.rawdata.build_workbooks sets RAW_ROOT.
"""

import os
import re
from glob import glob

import numpy as np
import pandas as pd

from . import env

logger = env.logger

DATASET_NAME = "D3_mustafa"
# folder of the raw files (set by eis.rawdata.build_workbooks)
RAW_ROOT = os.path.join(env.DIR_DATASETS, "raw", "D3")
OUTPUT_PATH = os.path.join(env.DIR_DATASETS, "rebuilt", f"{DATASET_NAME}_dataset.xlsx")

SOCS = list(range(5, 101, 5))  # 5, 10, 15, ..., 100

FNAME_RE = re.compile(r"^Hk_IFR14500_SoC_(?P<soc>\d+)_.*\.csv$")


def gather_files_for_soc(soc: int) -> list[str]:
    """All files of one state of charge over the cells and both test runs."""
    pattern = os.path.join(
        RAW_ROOT, "B*", "EIS measurements", "Test_*", "Hioki",
        f"Hk_IFR14500_SoC_{soc}_*.csv",
    )
    return sorted(glob(pattern))


def read_eis_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Frequency(Hz)": "f",
        "R(ohm)": "zr",
        "X(ohm)": "zj",
        "V(V)": "vdc",
        "T(deg C)": "temp",
        "Range(Ohm)": "range_ohm",
    })
    return df


def build_subset_for_soc(soc: int) -> tuple[pd.DataFrame, float, int]:
    """Mean spectrum of one state of charge, its label v_actual and the number of files."""
    files = gather_files_for_soc(soc)
    if not files:
        raise RuntimeError(f"No D3 files for SOC={soc}%")

    frames = []
    voltages = []
    for path in files:
        df = read_eis_file(path)
        voltages.append(float(df["vdc"].mean()))
        frames.append(df)

    cat = pd.concat(frames, ignore_index=True)
    agg = cat.groupby("f", sort=False)[["zr", "zj", "vdc"]].mean().reset_index()

    z = agg["zr"].to_numpy() + 1j * agg["zj"].to_numpy()
    agg["zmod"] = np.abs(z)
    agg["zphz"] = np.degrees(np.angle(z))
    agg["idc"] = np.nan
    agg = agg[["f", "zr", "zj", "zmod", "zphz", "idc", "vdc"]]
    agg = agg.sort_values("f", ascending=False).reset_index(drop=True)

    v_actual_cohort_mean = float(np.mean(voltages))
    return agg, v_actual_cohort_mean, len(files)


def build(output_path: str = OUTPUT_PATH) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # first pass: the spectrum and the label v_actual of every state of charge
    socinfo = []
    for soc in SOCS:
        spectrum, v_actual, n_files = build_subset_for_soc(soc)
        socinfo.append((soc, v_actual, n_files, spectrum))

    # sheet name v_nom: v_actual rounded to 4 decimals, raised by 1 mV while it is taken
    used_v_nom: set[float] = set()
    finalised = []
    for soc, v_actual, n_files, spectrum in socinfo:
        v_nom = round(v_actual, 4)
        while v_nom in used_v_nom:
            v_nom = round(v_nom + 0.001, 4)
        used_v_nom.add(v_nom)
        finalised.append((soc, v_nom, v_actual, n_files, spectrum))

    meta_rows = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for soc, v_nom, v_actual, n_files, spectrum in finalised:
            sheet = str(v_nom)
            spectrum.to_excel(writer, sheet_name=sheet, index=False)
            meta_rows.append({
                "v_nom": v_nom,
                "v_actual": v_actual,
                "n_points": int(len(spectrum)),
                "f_min": float(spectrum["f"].min()),
                "f_max": float(spectrum["f"].max()),
                "soc_pct": soc,
                "n_files_averaged": n_files,
            })
            logger.info(
                "  SOC %3d%%  ->  v_nom=%.4f V, v_actual=%.4f V, %2d pts, f in [%.3f, %.1f] Hz, n_files=%d",
                soc, v_nom, v_actual, len(spectrum),
                spectrum["f"].min(), spectrum["f"].max(), n_files,
            )

        meta = pd.DataFrame(meta_rows)
        meta.to_excel(writer, sheet_name="meta", index=False)

    logger.info("Wrote %s (%d sheets)", output_path, len(SOCS) + 1)
    return output_path
