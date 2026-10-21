"""Per-state circuit fits and the leave-one-state-out circuit mean.

The constants `CIRCUITS` and `FIT_KW` fix the circuit (ECM-W) and the settings of every
per-state fit. For one held-out state, `ecm_mean_loso` builds the circuit mean m = (m_re, m_im)
of K-K GP-ECM from the training states only.
"""

import time

import numpy as np
import pandas as pd

from . import ecm
from .ecm import predict_z

CIRCUITS: dict[str, type] = {"ECM-W": ecm.ECMW}
FIT_KW = dict(adam_steps=500, adam_lr=0.05, lbfgs_outer=5, lbfgs_max_iter=200)


def fit_state(cname: str, cls, group: pd.DataFrame) -> dict:
    """Fit one circuit to one spectrum (the rows of a single state).

    Returns the `fit_to_spectrum` result dict plus `fit_time_s` and `model`.
    """
    omega = group["omega"].to_numpy()
    z = (group["zr"] + 1j * group["zj"]).to_numpy()
    model = cls(ecm.init_from_spectrum(cname, omega, z))
    t0 = time.perf_counter()
    res = ecm.fit_to_spectrum(model, omega, z, **FIT_KW)
    res["fit_time_s"] = time.perf_counter() - t0
    res["model"] = model
    return res


def ecm_mean_loso(grid: dict, df: pd.DataFrame, held_out: float, cname: str, cls):
    """Circuit mean over the full grid when the state `held_out` is withheld.

    Each training state is fitted on its own full spectrum. The held-out state is never fitted.
    Its parameters are interpolated across the training states at its coordinate v
    (`ecm.interpolate_params`). Returns (mre, mim) over the full grid of `eis.gpexp.build_grid`.
    """
    G = grid["omega"].size
    mre = np.zeros(G); mim = np.zeros(G)
    fits = {}
    for v_nom, g in df.groupby("v_nom", sort=True):
        if float(v_nom) == float(held_out):
            continue                                    # the held-out state is never fitted
        res = fit_state(cname, cls, g)
        fits[float(v_nom)] = {"v": float(g["v"].iloc[0]), "params": res["params"]}
        rows = np.where(grid["v_nom"] == float(v_nom))[0]
        z = predict_z(res["model"], grid["omega"][rows])
        mre[rows] = z.real; mim[rows] = z.imag
    # held-out state: parameters interpolated across the training states
    v_train = np.array([fits[o]["v"] for o in fits])
    params_train = [fits[o]["params"] for o in fits]
    rows = np.where(grid["v_nom"] == float(held_out))[0]
    v_s = float(grid["v"][rows][0])
    model = cls(ecm.interpolate_params(cls, v_train, params_train, v_s))
    z = predict_z(model, grid["omega"][rows])
    mre[rows] = z.real; mim[rows] = z.imag
    return mre, mim
