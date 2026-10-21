"""K-K consistency of the predictions under leave-one-state-out: the sign errors and the K-K
residual rho of the held-out predictions, the imaginary-channel error floor of the constraint,
the two predictions at the D3 state v = 3.2261 and the optimizer restarts of GP-RBF-0 there.

The models and their settings are those of eis.loso: GP-RBF-0, GP-RBF-c, K-K GP-0 and K-K GP-ECM
(with the circuit ECM-W).

A held-out frequency with measured Im Z != 0 has a sign error when the sign of the predicted
Im Z differs from the sign of the measured Im Z. A state counts under "any" when it has at least
one sign error (any_wrong_sign), and under "majority" when more than half of its frequencies have
one (majority_wrong_sign).

With H the operator of eis.hilbert.hilbert_matrix on the frequencies of the held-out state and
m = (m_re, m_im) the mean of the model, the K-K residual is

    rho = ||(Im Z_pred - m_im) - H (Re Z_pred - m_re)|| / ||Im Z_pred - m_im||.

The residual rho comes from `sign_errors` for the two zero-mean models (m = 0), from
`rho_gp_rbf_c` for GP-RBF-c about its fitted constant mean m = (c_re, c_im) and from
`sign_errors_kkgp_ecm` for K-K GP-ECM about its circuit mean. Since H 1 = 0, the term H c_re is
zero. The construction of K-K GP-0 and K-K GP-ECM gives rho = 0 up to rounding.

For a mean m on the n frequencies of one state, every prediction of the constrained model lies in
the set

    A(m) = { (m_re + u, m_im + H u) : u in R^n }.

The distance from the measured spectrum y = (y_re, y_im) to A(m) bounds from below the error
(both channels together) of every constrained predictor with this mean. With r = y - m, the
closest point of A(m) solves

    min_u ||r_re - u||^2 + ||r_im - H u||^2,   that is   (I + H^T H) u = r_re + H^T r_im,

and e_re = r_re - u, e_im = r_im - H u are its errors in the two channels. The error floor is
e_im. A predictor has a smaller imaginary error only with a larger real error. The three means
of the constrained models are:
  zero      m = 0 (`floor_zero_or_ecm`).
  constant  m = (c_re, c_im) with the c that makes the distance smallest (`floor_constant`).
            H 1 = 0, so c_re is absorbed into u, and the best c_im removes the mean of
            r_im - H u. With the centring projector P = I - (1/n) 1 1^T:
                (I + H^T P H) u = y_re + H^T P y_im,   e_im = P (y_im - H u).
  ecm       the circuit mean of K-K GP-ECM with the state held out (eis.fitting.ecm_mean_loso,
            then `floor_zero_or_ecm`).
"""

import time

import numpy as np
import pandas as pd
import torch

from . import data, datasets, fitting, gp, hilbert
from .fitting import ecm_mean_loso
from .gpexp import build_grid, FIT_KW
from .utils import rmse

# model label -> class, constructor arguments and zero-mean flag (the settings of eis.loso)
MODELS = {
    "GP-RBF-0": dict(cls="GPRBF", kw=dict(mean="zero"), zero_mean=True),
    "GP-RBF-c": dict(cls="GPRBF", kw=dict(mean="const"), zero_mean=False),
    "K-K GP-0": dict(cls="KKGP", kw=dict(kernel="rbf", mean="M2"), zero_mean=True),
}

CIRCUIT = "ECM-W"   # the circuit of K-K GP-ECM (eis.ecm.ECMW)

# the D3 state of signflip_arrays and restart_rows
STATE = 3.2261


def sign_errors(dname: str, held_out: float, label: str, df: pd.DataFrame) -> dict:
    """Fit the model `label` on all states except `held_out`.

    Returns one row with the sign errors and the K-K residual rho at the held-out state.
    """
    torch.set_num_threads(1)
    spec = MODELS[label]
    grid = build_grid(df)
    G = grid["omega"].size
    v_nom = grid["v_nom"]
    states = sorted(set(v_nom))
    v_by_state = {s: float(grid["v"][v_nom == s][0]) for s in states}
    v_min, v_max = min(v_by_state.values()), max(v_by_state.values())

    mask = v_nom != held_out
    # the held-out state is not in the training set, and every other state is
    if mask[v_nom == held_out].any():
        raise AssertionError(f"LEAK: held-out state {held_out} in train ({dname})")
    if not mask[v_nom != held_out].all():
        raise AssertionError(f"LEAK: a training state dropped ({dname},{held_out})")

    model = (gp.GPRBF(**spec["kw"]) if spec["cls"] == "GPRBF"
             else gp.KKGP(**spec["kw"]))
    model.set_data(grid["omega"], grid["state_id"], grid["feat"],
                   grid["obs_re"], grid["obs_im"], np.zeros(G), np.zeros(G), mask)
    t0 = time.perf_counter()
    model.fit(**FIT_KW)
    fit_time = time.perf_counter() - t0
    pred = model.predict(observation_noise=True)

    te = np.where(~mask)[0]
    y_im = grid["obs_im"][te]
    mu_re, mu_im = np.asarray(pred["mu_re"]), np.asarray(pred["mu_im"])

    # sign errors
    # a measured Im Z of exactly 0 has no sign (the datasets have no such point)
    live = y_im != 0.0
    wrong = (np.sign(mu_im) != np.sign(y_im)) & live
    n_live = int(live.sum())
    n_wrong = int(wrong.sum())
    # largest |predicted Im Z| among the frequencies with a sign error
    worst = float(np.abs(mu_im[wrong]).max()) if n_wrong else 0.0

    # K-K residual rho (zero-mean models only)
    kk_rel = np.nan
    if spec["zero_mean"]:
        # the test rows are one state, frequencies ascending
        omega_s = grid["omega"][te]
        H = hilbert.hilbert_matrix(omega_s)
        resid = mu_im - H @ mu_re
        denom = np.linalg.norm(mu_im)
        kk_rel = float(np.linalg.norm(resid) / denom) if denom > 0 else np.nan

    v_s = v_by_state[held_out]
    return {
        "dataset": dname, "method": label,
        "held_out_v_nom": float(held_out), "held_out_v": v_s,
        "region": "boundary" if (v_s == v_min or v_s == v_max) else "interior",
        "n_test": int(te.size), "n_live": n_live,
        "n_wrong_sign": n_wrong,
        "frac_wrong_sign": n_wrong / n_live if n_live else np.nan,
        "any_wrong_sign": bool(n_wrong > 0),
        "majority_wrong_sign": bool(n_live and n_wrong > n_live / 2),
        "worst_wrong_im": worst,
        "kk_rel_resid": kk_rel,
        "rmse_im": float(np.sqrt(np.mean((y_im - mu_im) ** 2))),
        "fit_time_s": fit_time,
    }


def sign_errors_kkgp_ecm(dname: str, held_out: float, df: pd.DataFrame) -> dict:
    """Fit K-K GP-ECM on all states except `held_out`.

    Returns one row with the sign errors and the K-K residual rho about the circuit mean at the
    held-out state.
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

    cls = fitting.CIRCUITS[CIRCUIT]
    mre, mim = ecm_mean_loso(grid, df, held_out, CIRCUIT, cls)

    model = gp.KKGP(kernel="rbf", mean="M1")
    model.set_data(grid["omega"], grid["state_id"], grid["feat"],
                   grid["obs_re"], grid["obs_im"], mre, mim, mask)
    t0 = time.perf_counter()
    model.fit(**FIT_KW)
    fit_time = time.perf_counter() - t0
    pred = model.predict(observation_noise=True)

    te = np.where(~mask)[0]
    y_im = grid["obs_im"][te]
    mu_re, mu_im = np.asarray(pred["mu_re"]), np.asarray(pred["mu_im"])

    # sign errors: predicted Im Z against measured Im Z
    live = y_im != 0.0
    wrong = (np.sign(mu_im) != np.sign(y_im)) & live
    n_live = int(live.sum()); n_wrong = int(wrong.sum())
    worst = float(np.abs(mu_im[wrong]).max()) if n_wrong else 0.0

    # K-K residual rho about the circuit mean
    omega_s = grid["omega"][te]
    H = hilbert.hilbert_matrix(omega_s)
    r_re = mu_re - mre[te]
    r_im = mu_im - mim[te]
    denom = np.linalg.norm(r_im)
    kk_rel = float(np.linalg.norm(r_im - H @ r_re) / denom) if denom > 0 else np.nan

    v_s = v_by_state[held_out]
    return {
        "dataset": dname, "method": "K-K GP-ECM",
        "held_out_v_nom": float(held_out), "held_out_v": v_s,
        "region": "boundary" if (v_s == v_min or v_s == v_max) else "interior",
        "n_test": int(te.size), "n_live": n_live,
        "n_wrong_sign": n_wrong,
        "frac_wrong_sign": n_wrong / n_live if n_live else np.nan,
        "any_wrong_sign": bool(n_wrong > 0),
        "majority_wrong_sign": bool(n_live and n_wrong > n_live / 2),
        "worst_wrong_im": worst,
        "kk_rel_resid": kk_rel,
        "rmse_im": float(np.sqrt(np.mean((y_im - mu_im) ** 2))),
        "fit_time_s": fit_time,
    }


def rho_gp_rbf_c(dname: str, held_out: float, df: pd.DataFrame) -> dict:
    """Fit GP-RBF-c on all states except `held_out`.

    Returns one row with the fitted constant mean (c_re, c_im) and the K-K residual rho about it
    (kk_rel_resid_const_frame).
    """
    torch.set_num_threads(1)
    grid = build_grid(df)
    G = grid["omega"].size
    v_nom = grid["v_nom"]
    mask = v_nom != held_out
    if mask[v_nom == held_out].any():
        raise AssertionError(f"LEAK: held-out state {held_out} in train ({dname})")

    model = gp.GPRBF(mean="const")
    model.set_data(grid["omega"], grid["state_id"], grid["feat"],
                   grid["obs_re"], grid["obs_im"], np.zeros(G), np.zeros(G), mask)
    model.fit(**FIT_KW)
    pred = model.predict(observation_noise=True)

    with torch.no_grad():
        c_re, c_im = (float(model.c[0]), float(model.c[1]))

    te = np.where(~mask)[0]
    mu_re, mu_im = np.asarray(pred["mu_re"]), np.asarray(pred["mu_im"])
    H = hilbert.hilbert_matrix(grid["omega"][te])
    r_re, r_im = mu_re - c_re, mu_im - c_im
    denom = np.linalg.norm(r_im)
    rho = float(np.linalg.norm(r_im - H @ r_re) / denom) if denom > 0 else np.nan
    return {"dataset": dname, "method": "GP-RBF-c",
            "held_out_v_nom": float(held_out), "c_re": c_re, "c_im": c_im,
            "kk_rel_resid_const_frame": rho}


def floor_zero_or_ecm(H, r_re, r_im):
    """Errors (e_re, e_im) of the point of A(m) closest to y, from r = (r_re, r_im) = y - m."""
    n = H.shape[0]
    A = np.eye(n) + H.T @ H
    u = np.linalg.solve(A, r_re + H.T @ r_im)
    return r_re - u, r_im - H @ u


def floor_constant(H, y_re, y_im):
    """Errors (e_re, e_im) for the constant mean that brings A(m) closest to y = (y_re, y_im)."""
    n = H.shape[0]
    P = np.eye(n) - np.ones((n, n)) / n          # centring projector
    A = np.eye(n) + H.T @ P @ H
    u = np.linalg.solve(A, y_re + H.T @ P @ y_im)
    return y_re - u, P @ (y_im - H @ u)


def error_floor_rows() -> pd.DataFrame:
    """Error floors of the three means at every state of the three datasets.

    One row per (dataset, state, mean): held_out_v, region, n, floor_im_rms and floor_re_rms
    (RMS of e_im and e_re), yim_rms and yim_absmed (RMS and median of |y_im|), and
    n_floor_exceeds (frequencies with |e_im| > |y_im|). The mean of a row is "zero",
    "constant" or "ecm".
    """
    rows = []
    for dname in datasets.DATASETS:
        datasets.resolve(dname)
        df = data.load_dataset(dname)
        grid = build_grid(df)
        cls = fitting.CIRCUITS[CIRCUIT]
        states = sorted(df["v_nom"].unique())
        v_by = {s: float(grid["v"][grid["v_nom"] == s][0]) for s in states}
        vmin, vmax = min(v_by.values()), max(v_by.values())

        for s in states:
            idx = np.where(grid["v_nom"] == float(s))[0]
            om = grid["omega"][idx]
            y_re, y_im = grid["obs_re"][idx], grid["obs_im"][idx]
            H = hilbert.hilbert_matrix(om)
            n = idx.size

            mre, mim = ecm_mean_loso(grid, df, float(s), CIRCUIT, cls)
            mre_s, mim_s = mre[idx], mim[idx]

            cases = {
                "zero":     floor_zero_or_ecm(H, y_re, y_im),
                "constant": floor_constant(H, y_re, y_im),
                "ecm":      floor_zero_or_ecm(H, y_re - mre_s, y_im - mim_s),
            }
            for mean, (e_re, e_im) in cases.items():
                rows.append({
                    "dataset": dname, "held_out_v": v_by[s], "mean": mean,
                    "region": "boundary" if v_by[s] in (vmin, vmax) else "interior",
                    "n": n,
                    "floor_im_rms": float(np.sqrt(np.mean(e_im ** 2))),
                    "floor_re_rms": float(np.sqrt(np.mean(e_re ** 2))),
                    "yim_rms": float(np.sqrt(np.mean(y_im ** 2))),
                    "yim_absmed": float(np.median(np.abs(y_im))),
                    # frequencies where |e_im| > |y_im|
                    "n_floor_exceeds": int((np.abs(e_im) > np.abs(y_im)).sum()),
                })

    out = pd.DataFrame(rows)
    return out


def restart_rows() -> pd.DataFrame:
    """Restarts of the GP-RBF-0 fit at the D3 state v = 3.2261.

    GP-RBF-0 is fitted as in eis.loso.fit_gp_rbf with this state held out, from nine starting
    points: the default start (all length scales 1) and the length scale of v set to 0.005, 0.01,
    0.02, 0.05, 0.1, 0.2, 0.5 and 1.0 in both channels. Each row holds one start with the final
    negative log marginal likelihood (nll_final), the RMSE of Im Z at the held-out state (rmse_im),
    the number of frequencies with predicted Im Z > 0 (n_im_positive) and the fitted length scale
    of v in the imaginary channel (ls_v_im). Every measured Im Z of this state is negative.
    """
    df3 = data.load_dataset("D3_mustafa")
    g3 = build_grid(df3)
    G3 = g3["omega"].size
    STATE = 3.2261
    mask3 = g3["v_nom"].astype(float) != STATE
    rows = np.where(~mask3)[0]
    zi = g3["obs_im"][rows]

    results = []
    # default start of GPRBF: log_ls = 0, so every length scale is 1
    inits = [("default", None)]
    for lsv in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        inits.append((f"ls_v={lsv}", lsv))

    for name, lsv in inits:
        mm = gp.GPRBF(mean="zero")
        if lsv is not None:
            with torch.no_grad():
                mm.log_ls[:, 1] = np.log(lsv)      # starting length scale of v, both channels
        mm.set_data(g3["omega"], g3["state_id"], g3["feat"], g3["obs_re"],
                    g3["obs_im"], np.zeros(G3), np.zeros(G3), mask3)
        info = mm.fit(**FIT_KW)
        pp = mm.predict(observation_noise=False)
        r_im = rmse(zi, pp["mu_im"])
        n_pos = int((pp["mu_im"] > 0).sum())
        ls_v_im = float(np.exp(mm.log_ls.detach().numpy()[1, 1]))
        results.append((name, info["nll_final"], r_im, n_pos, ls_v_im))
    return pd.DataFrame(results,
                        columns=["start", "nll_final", "rmse_im", "n_im_positive", "ls_v_im"])


def signflip_arrays() -> dict:
    """Predictions of K-K GP-0 and GP-RBF-0 at the D3 state v = STATE.

    Both models have the zero mean and are fitted on every other D3 state with the settings of
    eis.loso. Returns their predicted means (without the noise) at the 28 frequencies of the
    held-out state together with the measured spectrum: s (the state v_nom), logf (log10 f,
    ascending), zr and zi (measured Re Z and Im Z), kk_re and kk_im (K-K GP-0), gp_re and gp_im
    (GP-RBF-0), in ohm.
    """
    datasets.resolve("D3_mustafa")
    df = data.load_dataset("D3_mustafa")
    grid = build_grid(df)
    G = grid["omega"].size
    rows = np.where(grid["v_nom"].astype(float) == STATE)[0]
    rows = rows[np.argsort(grid["omega"][rows])]
    logf = np.log10(grid["omega"][rows] / (2.0 * np.pi))
    zr, zi = grid["obs_re"][rows], grid["obs_im"][rows]
    mask = grid["v_nom"].astype(float) != STATE       # training states: every other state

    def predict(model):
        model.set_data(grid["omega"], grid["state_id"], grid["feat"],
                       grid["obs_re"], grid["obs_im"], np.zeros(G), np.zeros(G), mask)
        model.fit(**FIT_KW)
        pf = model.predict_full(observation_noise=False)
        return pf["mu_re"][rows], pf["mu_im"][rows]

    kk_re, kk_im = predict(gp.KKGP(kernel="rbf", mean="M2"))   # K-K GP-0
    gp_re, gp_im = predict(gp.GPRBF(mean="zero"))              # GP-RBF-0
    return dict(s=STATE, logf=logf, zr=zr, zi=zi,
                kk_re=kk_re, kk_im=kk_im, gp_re=gp_re, gp_im=gp_im)
