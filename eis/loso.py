"""One held-out state under leave-one-state-out: GP-RBF-0 and GP-RBF-c, K-K GP-0, K-K GP-ECM and
ECM (circuit).

Each function fits one model with the state `held_out` withheld and scores the held-out spectrum.
The training mask is (v_nom != held_out), so the model is fitted on every point of the other
states of the dataset and predicts the full spectrum of the held-out state. Each function raises an
AssertionError ("LEAK: ...") if the mask holds a point of the held-out state or misses a point of
another state. The held-out values enter neither a circuit fit nor the likelihood nor the
posterior. Only the frequencies and the coordinate v of the held-out state are used. A state
with the smallest or the largest v of its dataset is labelled "boundary", and every other state
"interior". The fits use no random numbers.

The models:
  GP-RBF-0, GP-RBF-c  eis.gp.GPRBF, one RBF GP per channel (Re, Im) on (log omega, v) with no
                      coupling between the channels and no circuit mean. The mean is zero
                      (GP-RBF-0) or a fitted constant per channel (GP-RBF-c).
  K-K GP-0            eis.gp.KKGP(kernel="rbf", mean="M2") with the mean m = 0: the GP on the
                      full grid with the RBF kernel on (log omega, v), whose imaginary channel is
                      H_G times the real channel.
  K-K GP-ECM          eis.gp.KKGP(kernel="rbf", mean="M1"). The mean m = (m_re, m_im) is the
                      circuit impedance Z_ECM of eis.ecm.ECMW, and the residual obeys
                      Im - m_im = H_G (Re - m_re). The circuit mean comes from
                      eis.fitting.ecm_mean_loso: each training state is fitted on its own full
                      spectrum, and the held-out state receives the parameters interpolated
                      across the training states at its v (linear in log or logit space, clamped
                      at the ends).
  ECM (circuit)       the interpolated circuit alone, scored by fit_kkgp_ecm in the columns
                      ecm_rmse_re and ecm_rmse_im.

Each function returns one table row: the dataset, the method label, the held-out state (v_nom
and v), its region, n_train_states, the test metrics of eis.gpexp.case_metrics (n_test, RMSE
Re/Im, 95% coverage per channel and pooled, interval width, NLL), the fit time in seconds and the
fitted hyperparameters. The row of fit_kkgp_ecm also holds the circuit name, ecm_rmse_re and
ecm_rmse_im, and gp_corr_re and gp_corr_im (the largest |GP prediction - circuit mean| over the
held-out frequencies).
"""

import time

import numpy as np
import pandas as pd
import torch

from . import fitting, gp, utils
from .fitting import ecm_mean_loso
from .gpexp import build_grid, case_metrics, FIT_KW

METHOD_KKGP_ZERO = "KKGP-zeromean"      # K-K GP-0
METHOD_KKGP_ECM = "KKGP-ecmmean"        # K-K GP-ECM


def fit_gp_rbf(dname: str, held_out: float, vname: str, mean: str, df: pd.DataFrame):
    """Fit GP-RBF on all states except `held_out` and score the held-out state."""
    torch.set_num_threads(1)
    grid = build_grid(df)
    G = grid["omega"].size
    v_nom = grid["v_nom"]
    states = sorted(set(v_nom))
    v_by_state = {s: float(grid["v"][v_nom == s][0]) for s in states}
    v_min, v_max = min(v_by_state.values()), max(v_by_state.values())

    mask = v_nom != held_out                      # training points: every point of the other states
    # the held-out state has no training point, and every other state is a training state
    if mask[v_nom == held_out].any():
        raise AssertionError(f"LEAK: held-out state {held_out} in train ({dname})")
    if not mask[v_nom != held_out].all():
        raise AssertionError(f"LEAK: a training state dropped ({dname},{held_out})")

    model = gp.GPRBF(mean=mean)
    model.set_data(grid["omega"], grid["state_id"], grid["feat"],
                   grid["obs_re"], grid["obs_im"], np.zeros(G), np.zeros(G), mask)
    t0 = time.perf_counter()
    info = model.fit(**FIT_KW)
    fit_time = time.perf_counter() - t0
    pred = model.predict(observation_noise=True)

    m = case_metrics(grid, mask, pred)
    v_s = v_by_state[held_out]
    region = "boundary" if (v_s == v_min or v_s == v_max) else "interior"
    sf, ls, nz = info["sigma_f"], info["lengthscales"], info["noise"]
    return {
        "dataset": dname, "method": vname, "split": "cross_state",
        "held_out_v_nom": float(held_out), "held_out_v": v_s, "region": region,
        "n_train_states": len(states) - 1, **m, "fit_time_s": fit_time,
        "nll_init": info["nll_init"], "nll_final": info["nll_final"],
        "sf_re": sf[0], "sf_im": sf[1], "noise_re": nz[0], "noise_im": nz[1],
        "ls_re_logw": ls[0][0], "ls_re_v": ls[0][1],
        "ls_im_logw": ls[1][0], "ls_im_v": ls[1][1],
    }


def fit_kkgp_zero(dname: str, held_out: float, df: pd.DataFrame):
    """Fit the zero-mean K-K GP on all states except `held_out` and score the held-out state."""
    torch.set_num_threads(1)
    grid = build_grid(df)
    G = grid["omega"].size
    v_nom = grid["v_nom"]
    states = sorted(set(v_nom))
    v_by_state = {s: float(grid["v"][v_nom == s][0]) for s in states}
    v_min, v_max = min(v_by_state.values()), max(v_by_state.values())

    mask = v_nom != held_out                      # training points: every point of the other states
    # the held-out state has no training point, and every other state is a training state
    if mask[v_nom == held_out].any():
        raise AssertionError(f"LEAK: held-out state {held_out} in train ({dname})")
    if not mask[v_nom != held_out].all():
        raise AssertionError(f"LEAK: a training state dropped ({dname},{held_out})")

    model = gp.KKGP(kernel="rbf", mean="M2")      # zero mean (label 'M2')
    model.set_data(grid["omega"], grid["state_id"], grid["feat"],
                   grid["obs_re"], grid["obs_im"], np.zeros(G), np.zeros(G), mask)
    t0 = time.perf_counter()
    info = model.fit(**FIT_KW)
    fit_time = time.perf_counter() - t0
    pred = model.predict(observation_noise=True)

    m = case_metrics(grid, mask, pred)
    v_s = v_by_state[held_out]
    region = "boundary" if (v_s == v_min or v_s == v_max) else "interior"
    ls = info["lengthscales"]                      # [log omega, v]
    return {
        "dataset": dname, "method": METHOD_KKGP_ZERO, "split": "cross_state",
        "held_out_v_nom": float(held_out), "held_out_v": v_s, "region": region,
        "n_train_states": len(states) - 1, **m, "fit_time_s": fit_time,
        "nll_init": info["nll_init"], "nll_final": info["nll_final"],
        "sigma_f": info["sigma_f"], "ls_logw": ls[0], "ls_v": ls[1],
        "noise_re": info["sigma_re"], "noise_im": info["sigma_im"],
    }


def fit_kkgp_ecm(dname: str, held_out: float, cname: str, df: pd.DataFrame):
    """Fit K-K GP-ECM (mean label M1, circuit `cname`) on all states except `held_out`.

    Scores the held-out state with K-K GP-ECM and with the interpolated circuit alone.
    """
    torch.set_num_threads(1)
    grid = build_grid(df)
    v_nom = grid["v_nom"]
    states = sorted(set(v_nom))
    v_by_state = {s: float(grid["v"][v_nom == s][0]) for s in states}
    v_min, v_max = min(v_by_state.values()), max(v_by_state.values())

    mask = v_nom != held_out
    if mask[v_nom == held_out].any():
        raise AssertionError(f"LEAK: held-out state {held_out} in train ({dname})")
    if not mask[v_nom != held_out].all():
        raise AssertionError(f"LEAK: a training state dropped ({dname},{held_out})")

    cls = fitting.CIRCUITS[cname]
    mre, mim = ecm_mean_loso(grid, df, held_out, cname, cls)

    model = gp.KKGP(kernel="rbf", mean="M1")
    model.set_data(grid["omega"], grid["state_id"], grid["feat"],
                   grid["obs_re"], grid["obs_im"], mre, mim, mask)
    t0 = time.perf_counter()
    info = model.fit(**FIT_KW)
    fit_time = time.perf_counter() - t0
    pred = model.predict(observation_noise=True)

    te = np.where(~mask)[0]
    m = case_metrics(grid, mask, pred)
    v_s = v_by_state[held_out]
    region = "boundary" if (v_s == v_min or v_s == v_max) else "interior"
    ls = info["lengthscales"]
    return {
        "dataset": dname, "method": METHOD_KKGP_ECM, "circuit": cname, "split": "cross_state",
        "held_out_v_nom": float(held_out), "held_out_v": v_s, "region": region,
        "n_train_states": len(states) - 1, **m,
        "ecm_rmse_re": utils.rmse(grid["obs_re"][te], mre[te]),
        "ecm_rmse_im": utils.rmse(grid["obs_im"][te], mim[te]),
        "gp_corr_re": float(np.abs(pred["mu_re"] - mre[te]).max()),
        "gp_corr_im": float(np.abs(pred["mu_im"] - mim[te]).max()),
        "fit_time_s": fit_time, "nll_init": info["nll_init"],
        "nll_final": info["nll_final"], "sigma_f": info["sigma_f"],
        "ls_logw": ls[0], "ls_v": ls[1],
        "noise_re": info["sigma_re"], "noise_im": info["sigma_im"],
    }
