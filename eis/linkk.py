"""The Lin-KK test of the measured spectra (`linKK` of the package impedance).

Lin-KK (Schoenleber et al., 2014) fits each spectrum with a model whose every term satisfies the
K-K relation,

    Z_fit(omega) = R_0 + sum_{k=1..M} R_k / (1 + j omega tau_k) + j omega L + 1 / (j omega C),

with tau_k fixed and log-spaced from 1 / (2 pi f_max) to 1 / (2 pi f_min). R_0, R_k, L and 1/C
solve a linear least-squares problem in the residuals divided by |Z| (with fit_type="complex",
the real and imaginary parts together). The term 1 / (j omega C) is present with add_cap=True.
The relative residuals are

    Delta_re = (Re Z - Re Z_fit) / |Z|,   Delta_im = (Im Z - Im Z_fit) / |Z|,
    |Delta| = sqrt(Delta_re^2 + Delta_im^2).

For a spectrum with n frequencies, `run_dataset` makes two fits.
  - The automatic fit increases M = 1, 2, ... until
    mu = 1 - sum_{R_k < 0} |R_k| / sum_{R_k >= 0} |R_k| is at most the cutoff c, or until
    M = max(5, n - 1).
  - The matched fit uses M = min(M_MANUAL, n - 2) without the cutoff.
Each spectrum is tested on its own.
"""

import numpy as np
import pandas as pd
import impedance.validation as _val

from . import data, datasets, env

logger = env.logger


def _patch_impedance():
    """Make `linKK` of impedance 1.7.1 run with numpy 2.

    `eval_linKK` writes the fitted circuit as a string and evaluates it with
    `impedance.validation.circuit_elements` as globals. With numpy 2 the string holds
    `np.float64(...)`, and that dict has no name `np`, so every call of `linKK` raises NameError.
    The patch replaces the dict by a copy with `np` added. The fitted values do not change.
    """
    if "np" not in _val.circuit_elements:
        _val.circuit_elements = {**_val.circuit_elements, "np": np}


_patch_impedance()
linKK = _val.linKK


def linkk_state(f, Z, c, max_M, fit_type, add_cap):
    """Lin-KK fit of one spectrum.

    Returns M, mu, Z_fit, the relative residuals res_re = Delta_re and res_im = Delta_im as
    returned by `linKK`, rel_cmplx = |Delta| and abs_resid = |Z - Z_fit|.
    """
    M, mu, Z_fit, res_re, res_im = linKK(
        f, Z, c=c, max_M=max_M, fit_type=fit_type, add_cap=add_cap)
    abs_resid = np.abs(Z - Z_fit)                      # |Z - Z_fit|
    rel_cmplx = np.sqrt(res_re ** 2 + res_im ** 2)     # |Delta|
    return dict(M=int(M), mu=float(mu), Z_fit=Z_fit,
                res_re=np.asarray(res_re, float), res_im=np.asarray(res_im, float),
                rel_cmplx=rel_cmplx, abs_resid=abs_resid)


M_MANUAL = 20      # M of the matched fit (the automatic rule picks 16 to 24 on D1 and D3)


def per_state_metrics(dname, v_nom, v, f, Z, r, rm):
    """Metrics of one state. `r` is the fit with automatic M and `rm` the fit with matched M.

    The columns *_cap use the points with Im Z < 0 only.
    """
    cap = Z.imag < 0.0                                 # points with Im Z < 0
    def med(x, m=None):
        x = x if m is None else x[m]
        return float(np.median(x)) if x.size else np.nan
    def rms(x, m=None):
        x = x if m is None else x[m]
        return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan
    return {
        "dataset": dname, "v_nom": float(v_nom), "v": float(v),
        "n_freq": int(f.size), "f_min": float(f.min()), "f_max": float(f.max()),
        "n_inductive": int((Z.imag > 0).sum()), "frac_inductive": float((Z.imag > 0).mean()),
        "M": r["M"], "mu": r["mu"],
        # relative residuals, all points
        "med_rel_re": med(np.abs(r["res_re"])), "med_rel_im": med(np.abs(r["res_im"])),
        "med_rel_cmplx": med(r["rel_cmplx"]), "max_rel_cmplx": float(r["rel_cmplx"].max()),
        "rms_rel_re": rms(r["res_re"]), "rms_rel_im": rms(r["res_im"]),
        # relative residuals at the points with Im Z < 0
        "med_rel_cmplx_cap": med(r["rel_cmplx"], cap),
        "max_rel_cmplx_cap": float(np.max(r["rel_cmplx"][cap])) if cap.any() else np.nan,
        # absolute residual |Z - Z_fit|
        "med_abs_cmplx": med(r["abs_resid"]), "max_abs_cmplx": float(r["abs_resid"].max()),
        "zmod_med": float(np.median(np.abs(Z))),
        # fit with matched M
        "M_manual": rm["M"], "mu_manual": rm["mu"],
        "med_rel_cmplx_manual": med(rm["rel_cmplx"]), "max_rel_cmplx_manual": float(rm["rel_cmplx"].max()),
    }


def run_dataset(dname, c, fit_type, add_cap):
    """Make both fits on every state of `dname`.

    Returns the per-state rows and the per-frequency rows.
    """
    datasets.resolve(dname)
    df = data.load_dataset(dname)
    per_rows, res_rows = [], []
    for v_nom, g in df.groupby("v_nom", sort=True):
        g = g.sort_values("f")                         # ascending frequency
        f = g["f"].to_numpy(float)
        Z = g["zr"].to_numpy(float) + 1j * g["zj"].to_numpy(float)
        v = float(g["v"].iloc[0])
        max_M = int(max(5, f.size - 1))                # largest M of the automatic rule
        r = linkk_state(f, Z, c=c, max_M=max_M, fit_type=fit_type, add_cap=add_cap)
        # fit with matched M (c=None: exactly m_man elements)
        m_man = int(min(M_MANUAL, f.size - 2))
        rm = linkk_state(f, Z, c=None, max_M=m_man, fit_type=fit_type, add_cap=add_cap)
        per_rows.append(per_state_metrics(dname, v_nom, v, f, Z, r, rm))
        for i in range(f.size):
            res_rows.append({
                "dataset": dname, "v_nom": float(v_nom), "f": float(f[i]),
                "omega": float(2 * np.pi * f[i]), "zr": float(Z[i].real), "zj": float(Z[i].imag),
                "zfit_re": float(r["Z_fit"][i].real), "zfit_im": float(r["Z_fit"][i].imag),
                "res_re": float(r["res_re"][i]), "res_im": float(r["res_im"][i]),
                "rel_cmplx": float(r["rel_cmplx"][i]), "abs_cmplx": float(r["abs_resid"][i]),
            })
        logger.info("  %s v_nom=%-7s M=%d mu=%.3f med|Delta|=%.4f (cap %.4f) | manual M=%d med|Delta|=%.4f",
                    dname, f"{float(v_nom):g}", r["M"], r["mu"],
                    per_rows[-1]["med_rel_cmplx"], per_rows[-1]["med_rel_cmplx_cap"],
                    per_rows[-1]["M_manual"], per_rows[-1]["med_rel_cmplx_manual"])
    return per_rows, res_rows


def summarize(per):
    """Return one row per dataset with the medians of the per-state metrics over its states.

    Each row also gives the state with the largest med_rel_cmplx (worst_state_v_nom and
    worst_med_rel_cmplx).
    """
    rows = []
    for dname, g in per.groupby("dataset", sort=True):
        rows.append({
            "dataset": dname, "n_states": int(len(g)),
            "med_M": float(np.median(g["M"])), "med_mu": float(np.median(g["mu"])),
            # median over the states of the per-state median |Delta|
            "med_rel_cmplx": float(np.median(g["med_rel_cmplx"])),
            "med_rel_re": float(np.median(g["med_rel_re"])),
            "med_rel_im": float(np.median(g["med_rel_im"])),
            "med_rel_cmplx_cap": float(np.median(g["med_rel_cmplx_cap"])),
            "med_abs_cmplx": float(np.median(g["med_abs_cmplx"])),
            # fit with matched M
            "med_M_manual": float(np.median(g["M_manual"])),
            "med_rel_cmplx_manual": float(np.median(g["med_rel_cmplx_manual"])),
            "worst_state_v_nom": float(g.loc[g["med_rel_cmplx"].idxmax(), "v_nom"]),
            "worst_med_rel_cmplx": float(g["med_rel_cmplx"].max()),
        })
    return pd.DataFrame(rows)
