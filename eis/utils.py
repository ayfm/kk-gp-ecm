"""Error metrics and helpers of the notebooks: parallel fits, output files and table layout."""

import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# rows of the result tables: the interior and the boundary states of each dataset
REGIONS = [("D1_zhang", "interior"), ("D1_zhang", "boundary"), ("D2_rashid", "interior"),
           ("D2_rashid", "boundary"), ("D3_mustafa", "interior"), ("D3_mustafa", "boundary")]


# helpers of the notebooks

def held_out_cases(tables: dict) -> list:
    """Return (dataset, held-out state, table) for every state of every dataset in `tables`."""
    return [(name, float(s), df) for name, df in tables.items()
            for s in sorted(df["v_nom"].unique())]


def fit_parallel(function, cases: list, n_jobs: int) -> pd.DataFrame:
    """Run function(*case) for every case in n_jobs worker processes.

    Returns the rows as one table and prints the number of fits and the run time.
    """
    start = time.perf_counter()
    rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(function)(*case) for case in cases)
    print(f"{len(rows)} fits in {(time.perf_counter() - start) / 60:.1f} min")
    return pd.DataFrame(rows)


def two_digits(x: float) -> str:
    """Text of x with two significant digits."""
    return f"{x:#.2g}"


def load_or_compute(path: str, compute, recompute: bool = False) -> pd.DataFrame:
    """Return the table of the CSV file `path`.

    If `recompute` is True or the file does not exist, the DataFrame returned by `compute()` is
    first written to `path`. The file is read with float_precision="round_trip", so every number
    read equals the number written.
    """
    if recompute or not os.path.isfile(path):
        compute().to_csv(path, index=False)
    return pd.read_csv(path, float_precision="round_trip")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nll_gaussian(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-12
) -> float:
    """Mean negative log predictive density under independent Gaussians N(mu, sigma^2)."""
    y_true = np.asarray(y_true)
    mu = np.asarray(mu)
    sigma = np.maximum(np.asarray(sigma), eps)
    var = sigma**2
    nll = 0.5 * (np.log(2.0 * np.pi * var) + (y_true - mu) ** 2 / var)
    return float(np.mean(nll))


def coverage(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float = 0.95
) -> float:
    """Fraction of y_true inside the central `level` interval of N(mu, sigma^2)."""
    from scipy.stats import norm

    y_true = np.asarray(y_true)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    z = norm.ppf(0.5 + level / 2.0)
    inside = np.abs(y_true - mu) <= z * sigma
    return float(np.mean(inside))
