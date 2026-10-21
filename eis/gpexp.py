"""Grid and metrics shared by the GP fits of eis.loso, eis.qgpdrt, eis.joint and eis.consistency.

  build_grid   : long-format DataFrame -> full grid for the GP (state-major, frequency
                 ascending) with the inputs feat = (log omega, v).
  case_metrics : test metrics of one fitted case (RMSE Re/Im, 95% coverage, width, NLL).

The constant `VARIANTS` names the two GP-RBF mean variants, and `FIT_KW` holds the fit settings
of every GP.
"""

import numpy as np
import pandas as pd

from . import utils

# the two unconstrained baselines: zero mean (GP-RBF-0) and fitted constant mean (GP-RBF-c)
VARIANTS = {"GP-RBF-0": "zero", "GP-RBF-c": "const"}

# norm.ppf(0.975): half-width factor of the central 95% interval.
Z95 = 1.959963984540054

# GP fit settings: Adam steps, then L-BFGS
FIT_KW = dict(adam_steps=300, adam_lr=0.05, lbfgs_outer=5, lbfgs_max_iter=100)


def build_grid(df: pd.DataFrame) -> dict:
    """Full grid of the GP: states in ascending v_nom, frequencies ascending within a state.

    Returns omega, state_id (an integer per state), v, feat = (log omega, v), obs_re = zr,
    obs_im = zj, the row key (v_nom, f) and v_nom per row.
    """
    g = df.sort_values(["v_nom", "f"], ascending=[True, True]).reset_index(drop=True)
    states = sorted(g["v_nom"].unique())
    sid = g["v_nom"].map({v: i for i, v in enumerate(states)}).to_numpy()
    omega = g["omega"].to_numpy(dtype=float)
    v = g["v"].to_numpy(dtype=float)
    feat = np.stack([np.log(omega), v], axis=1)
    return {
        "df": g, "omega": omega, "state_id": sid, "v": v, "feat": feat,
        "obs_re": g["zr"].to_numpy(dtype=float), "obs_im": g["zj"].to_numpy(dtype=float),
        "key": list(zip(g["v_nom"].to_numpy(), g["f"].to_numpy())),
        "v_nom": g["v_nom"].to_numpy(),
    }


def case_metrics(grid: dict, mask: np.ndarray, pred: dict) -> dict:
    """Test metrics of one fitted case. The standard deviations include the noise.

    The dict `pred` holds mu_re, mu_im, var_re, var_im at the test rows where(~mask) in that order
    (as returned by the GP `predict()`).
    """
    te = np.where(~mask)[0]
    yr, yi = grid["obs_re"][te], grid["obs_im"][te]
    mu_re, mu_im = pred["mu_re"], pred["mu_im"]
    sd_re = np.sqrt(np.maximum(pred["var_re"], 1e-300))
    sd_im = np.sqrt(np.maximum(pred["var_im"], 1e-300))
    m = {
        "n_test": int(te.size),
        "rmse_re": utils.rmse(yr, mu_re), "rmse_im": utils.rmse(yi, mu_im),
        "cov95_re": utils.coverage(yr, mu_re, sd_re), "cov95_im": utils.coverage(yi, mu_im, sd_im),
        "width_re": float(2 * Z95 * sd_re.mean()), "width_im": float(2 * Z95 * sd_im.mean()),
        "nll_re": utils.nll_gaussian(yr, mu_re, sd_re), "nll_im": utils.nll_gaussian(yi, mu_im, sd_im),
    }
    y = np.concatenate([yr, yi]); mu = np.concatenate([mu_re, mu_im])
    sd = np.concatenate([sd_re, sd_im])
    m["cov95_all"] = utils.coverage(y, mu, sd)
    m["nll_all"] = utils.nll_gaussian(y, mu, sd)
    return m
