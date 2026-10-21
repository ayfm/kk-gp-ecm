"""Loading of a dataset workbook into one long-format table."""

import os
from typing import Literal

import numpy as np
import pandas as pd

from . import datasets, env

logger = env.logger


# Columns of the table returned by load_dataset(). v_nom is the label of the state (the name of
# its sheet), v_actual is the state coordinate stored in the meta sheet, and v is whichever of
# the two `which_v` selects. f is the frequency in Hz, logf = log10 f, omega = 2 pi f,
# zr = Re Z and zj = Im Z in ohm, zmod = |Z|, zphz = phase of Z in degrees, idc and vdc as
# stored in the workbook.
DATASET_COLS = [
    "v_nom", "v_actual", "v",
    "f", "logf", "omega",
    "zr", "zj", "zmod", "zphz",
    "idc", "vdc",
]


WhichV = Literal["actual", "nominal"]


def load_dataset(
    dataset: str | None = None,
    which_v: WhichV = "actual",
    path: str | None = None,
) -> pd.DataFrame:
    """Load a dataset workbook as one long-format DataFrame.

    Give either `dataset` (a name in `eis.datasets.DATASETS`, for example "D1_zhang") or an
    explicit `path` to a workbook. The workbook has one sheet per state, named by its `v_nom`
    value, and a sheet `meta` with the columns `v_nom` and `v_actual`.

    Parameters
    ----------
    dataset : str, optional
        Registered dataset name, resolved to a workbook path by `datasets.resolve`.
    which_v : {"actual", "nominal"}
        Which value becomes the state coordinate `v`: `v_actual` ("actual") or `v_nom`
        ("nominal").
    path : str, optional
        Explicit workbook path. Takes precedence over `dataset`.

    Returns
    -------
    pandas.DataFrame
        Columns DATASET_COLS, sorted by v_nom ascending and f descending.
    """
    if path is None:
        if dataset is None:
            raise ValueError(
                "load_dataset requires either dataset=<name> (one of "
                f"{sorted(datasets.DATASETS)}) or an explicit path=."
            )
        path = datasets.resolve(dataset).xlsx_path
    elif not os.path.exists(path):
        raise FileNotFoundError(f"Dataset workbook not found at {path}.")

    xls = pd.ExcelFile(path, engine="openpyxl")
    if "meta" not in xls.sheet_names:
        raise ValueError(f"Workbook {path} is missing the required 'meta' sheet.")

    meta = pd.read_excel(xls, sheet_name="meta")
    # state coordinate of each state, keyed by the numeric v_nom value
    meta_v_actual = {float(r.v_nom): float(r.v_actual) for r in meta.itertuples()}

    # Sheets are matched to states by numeric value, so a state 3.4 is found in a sheet named
    # "3.4" or "3.40".
    sheet_by_value: dict[float, str] = {}
    for s in xls.sheet_names:
        if s == "meta":
            continue
        try:
            sheet_by_value[float(s)] = s
        except ValueError:
            continue  # ignore any non-numeric sheet

    # states listed in the meta sheet, in ascending order
    v_nom_values = sorted(meta_v_actual)

    frames = []
    for v_nom in v_nom_values:
        if v_nom not in sheet_by_value:
            raise ValueError(
                f"Workbook {path}: no data sheet for state v_nom={v_nom} "
                f"(numeric sheets present: {sorted(sheet_by_value)})."
            )
        df = pd.read_excel(xls, sheet_name=sheet_by_value[v_nom])
        df["v_nom"] = v_nom
        df["v_actual"] = meta_v_actual[v_nom]
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # derived columns
    out["logf"] = np.log10(out["f"])
    out["omega"] = 2.0 * np.pi * out["f"]
    out["v"] = out["v_actual"] if which_v == "actual" else out["v_nom"]

    out = out[DATASET_COLS].copy()
    out = out.sort_values(["v_nom", "f"], ascending=[True, False]).reset_index(drop=True)
    return out
