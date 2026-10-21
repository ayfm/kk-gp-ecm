"""The three datasets and the paths of their workbooks.

Each entry of `DATASETS` names one workbook in the folder `datasets/` of the repository
(`eis.env.DIR_DATASETS`). The loader eis.data.load_dataset and the notebooks in `notebooks/`
look the names up here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import env


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    xlsx_filename: str

    @property
    def xlsx_path(self) -> str:
        return os.path.join(env.DIR_DATASETS, self.xlsx_filename)


DATASETS: dict[str, DatasetSpec] = {
    "D1_zhang": DatasetSpec(
        name="D1_zhang",
        xlsx_filename="D1_zhang_dataset.xlsx",
    ),
    "D2_rashid": DatasetSpec(
        name="D2_rashid",
        xlsx_filename="D2_rashid_dataset.xlsx",
    ),
    "D3_mustafa": DatasetSpec(
        name="D3_mustafa",
        xlsx_filename="D3_mustafa_dataset.xlsx",
    ),
}


def resolve(name: str) -> DatasetSpec:
    """Look up a DatasetSpec by name. Raises ValueError for an unknown name and
    FileNotFoundError if its workbook is missing."""
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(f"Unknown dataset {name!r}. Known: {known}")
    spec = DATASETS[name]
    if not os.path.exists(spec.xlsx_path):
        raise FileNotFoundError(
            f"Dataset {name!r}: workbook not found at {spec.xlsx_path}. "
            f"See datasets/README.md for where the workbooks come from."
        )
    return spec
