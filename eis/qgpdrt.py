"""The model qGP-DRT of Maradesa, Py and Ciucci (2024) under leave-one-state-out.

The model is a GP over the distribution of relaxation times gamma of several states. The upstream
code is in quasigp/ without changes (see quasigp/SOURCE.md). Its quadrature matrices
`compute_A_re`, `compute_A_im` and its HMC sampler `generate_tmg` are used here with an
adaptation to leave-one-state-out.
  (1) The held-out state is one block of the latent vector with no observation rows, so its
      measured values enter no fit.
  (2) The latent has an inductance term L_s (the released upstream model has none).
  (3) The state coordinate and the impedance are standardized with the training states only.
  (4) The hyperparameters have bounds (among them sigma_n >= 1e-2 for the standardized noise
      level). The fit uses three fixed L-BFGS-B starts and a dense Cholesky solve, and each
      held-out state has a fixed HMC seed (`fold_seed`).

The model of state s, with omega = 2 pi f, is

    Z_s(f) = R_s + j omega L_s + integral of gamma_s(ln tau) / (1 + j omega tau) d ln tau,

discretized on T values of tau as Z_s = A_s x_s with the latent x_s = [R_s, L_s, gamma_s(1..T)]
(length T+2). With `with_L=False` the latent is [R_s, gamma_s(1..T)] (length T+1) as in the
upstream code. The notebooks use with_L=True (`VARIANTS`). States are stacked state-major. All
arrays are dense float64.

For one held-out state, `fit_qgpdrt` predicts the spectrum with `predict_loso` and scores it. The
predictive mean and standard deviation of each frequency are the moments of the HMC samples (the
standard deviation includes the noise). Its row holds the region, n_test, RMSE Re/Im, the 95%
coverage, interval width and NLL of the Gaussian with these moments, hmc_cov95_* (coverage of the
2.5 to 97.5 percentile interval of the HMC samples plus noise), T (number of tau values), n_reject
(rejected HMC proposals), seed (HMC seed), sigma_n, nmll, imp_scale_c (standard deviation of the
training impedance values) and fit_time_s.
"""

import io
import os
import sys
import time
from contextlib import redirect_stdout
from math import pi

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from . import data, datasets
from .gpexp import build_grid, case_metrics

TWO_PI = 2.0 * pi

# vendored upstream code in quasigp/. HMC.py imports repmat from numpy.matlib and never calls
# it. If numpy.matlib cannot be imported, a stand-in module provides repmat = np.tile.
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quasigp")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
try:
    import numpy.matlib  # noqa: F401
except Exception:
    from types import SimpleNamespace
    sys.modules.setdefault("numpy.matlib", SimpleNamespace(repmat=np.tile))
import qGP_DRT as _vqgp   # noqa: E402
import HMC as _vhmc       # noqa: E402


# hyperparameters (standardized units)
def theta_spec(with_L: bool):
    """Return (keys, bounds) of the hyperparameters of the L-BFGS-B fit (standardized units)."""
    # sigma_n >= 1e-2: the lower bound on the standardized noise level
    if with_L:
        keys = ["sigma_n", "sigma_R", "sigma_L", "sigma_k", "l_f", "l_psi", "l_R", "l_L"]
        bounds = [(1e-2, 1.0), (1e-2, 10.0), (1e-2, 10.0), (1e-2, 10.0),
                  (0.3, 3.0), (0.3, 5.0), (0.3, 5.0), (0.3, 5.0)]
    else:
        keys = ["sigma_n", "sigma_R", "sigma_k", "l_f", "l_psi", "l_R"]
        bounds = [(1e-2, 1.0), (1e-2, 10.0), (1e-2, 10.0),
                  (0.3, 3.0), (0.3, 5.0), (0.3, 5.0)]
    return keys, bounds


def block_dim(T: int, with_L: bool) -> int:
    return T + (2 if with_L else 1)


def gamma_start(with_L: bool) -> int:
    """Index of gamma_s(1) inside the latent block of one state."""
    return 2 if with_L else 1


# tau grid
def build_tau_grid(freq: np.ndarray, T: int):
    """T values of tau, uniform in log10(tau) on
    [log10(1 / (2 pi f_max)) - 0.5, log10(1 / (2 pi f_min)) + 0.5].

    Returns (tau, log_tau) with log_tau = ln(tau). The vendored `compute_A_*` take tau.
    """
    f_min, f_max = float(np.min(freq)), float(np.max(freq))
    lo = np.log10(1.0 / (TWO_PI * f_max)) - 0.5
    hi = np.log10(1.0 / (TWO_PI * f_min)) + 0.5
    tau = 10.0 ** np.linspace(lo, hi, T)
    return tau, np.log(tau)


# observation operator of one state (all states share one frequency grid)
def state_A(freq: np.ndarray, tau: np.ndarray, with_L: bool) -> np.ndarray:
    """Operator A_s of one state, shape (2F, T+2) with the inductance term and (2F, T+1) without.

    The rows are [Re Z at the F frequencies; Im Z at the F frequencies]. The columns are [1; 0]
    for R_s, [0; omega / omega_max] for L_s, and the vendored quadrature (compute_A_re;
    compute_A_im) for gamma_s. The factor 1 / omega_max keeps the column of L_s of order one,
    like the other columns, so the bounds of sigma_L in standardized units apply to it. The
    physical inductance is L_s / omega_max.
    """
    F = freq.size
    omega = TWO_PI * freq
    A_re = _vqgp.compute_A_re(freq, tau)   # (F, T)
    A_im = _vqgp.compute_A_im(freq, tau)   # (F, T), integrand -omega tau / (1 + (omega tau)^2)
    T = tau.size
    g0 = gamma_start(with_L)
    A = np.zeros((2 * F, T + g0))
    A[:F, 0] = 1.0                         # R_s: real part only
    if with_L:
        A[F:, 1] = omega / omega.max()     # L_s: imaginary part only, scaled by 1 / omega_max
    A[:F, g0:] = A_re                      # gamma_s: real part
    A[F:, g0:] = A_im                      # gamma_s: imaginary part
    return A


def blockdiag(A_state: np.ndarray, n_states: int) -> np.ndarray:
    """Block-diagonal matrix with n_states copies of A_state."""
    r, c = A_state.shape
    out = np.zeros((r * n_states, c * n_states))
    for k in range(n_states):
        out[r * k:r * (k + 1), c * k:c * (k + 1)] = A_state
    return out


# prior covariance of the latent
def assemble_Gamma(theta: dict, log_tau: np.ndarray, psi: np.ndarray, with_L: bool) -> np.ndarray:
    """Prior covariance Gamma of the stacked latent x = [x_1; ...; x_S].

    Three independent GP parts, each with a squared-exponential kernel in the standardized state
    coordinate psi: R (sigma_R, l_R), L (sigma_L, l_L) if with_L, and gamma (sigma_k, l_psi in
    psi times a squared-exponential kernel with l_f in ln tau).
    """
    S = psi.size
    T = log_tau.size
    bd = block_dim(T, with_L)
    g0 = gamma_start(with_L)
    D = bd * S

    dpsi2 = (psi[:, None] - psi[None, :]) ** 2                     # (S, S)
    KR = theta["sigma_R"] ** 2 * np.exp(-dpsi2 / (2 * theta["l_R"] ** 2))
    if with_L:
        KL = theta["sigma_L"] ** 2 * np.exp(-dpsi2 / (2 * theta["l_L"] ** 2))
    Kpsi_g = np.exp(-dpsi2 / (2 * theta["l_psi"] ** 2))            # (S, S)
    dlt = log_tau[:, None] - log_tau[None, :]
    Kf = np.exp(-(dlt ** 2) / (2 * theta["l_f"] ** 2))            # (T, T)
    sk2 = theta["sigma_k"] ** 2

    Gamma = np.zeros((D, D))
    for k in range(S):
        rk = bd * k
        for l in range(S):
            cl = bd * l
            Gamma[rk, cl] = KR[k, l]
            if with_L:
                Gamma[rk + 1, cl + 1] = KL[k, l]
            Gamma[rk + g0:rk + bd, cl + g0:cl + bd] = sk2 * Kpsi_g[k, l] * Kf
    return Gamma


# negative log marginal likelihood of the training states and the hyperparameter fit
def _chol(M):
    """Lower Cholesky factor of M (scipy cho_factor).

    If the factorization fails, the diagonal jitter eps * mean(diag M) is added with
    eps = 1e-9, 1e-8, ..., 1e-4 in turn. Raises LinAlgError if all of them fail.
    """
    d = M.shape[0]
    base = float(np.mean(np.diag(M)))
    for j in range(7):
        eps = 0.0 if j == 0 else (10.0 ** (-10 + j)) * base
        try:
            return cho_factor(M + eps * np.eye(d), lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("matrix not PD after escalating jitter")


def _nmll(theta_vec, keys, A_train, y, log_tau, psi_train, with_L):
    """Negative log marginal likelihood of y under N(0, Psi), Psi = A Gamma A^T + sigma_n^2 I.

    Returns 1e18 when Psi has no Cholesky factor.
    """
    theta = dict(zip(keys, theta_vec))
    Gamma = assemble_Gamma(theta, log_tau, psi_train, with_L)
    n = A_train.shape[0]
    Psi = A_train @ Gamma @ A_train.T + (theta["sigma_n"] ** 2) * np.eye(n)
    try:
        cf = _chol(Psi)
    except np.linalg.LinAlgError:
        return 1e18
    alpha = cho_solve(cf, y, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    return 0.5 * float(y @ alpha) + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)


def fit_theta(A_train, y, log_tau, psi_train, with_L, n_starts=3):
    """Minimize `_nmll` by L-BFGS-B within the bounds of `theta_spec`, from n_starts points.

    Start k sets every hyperparameter to lo + f_k (hi - lo), with f = linspace(0.25, 0.75,
    n_starts). Returns the best (theta dict, its negative log marginal likelihood).
    """
    keys, bounds = theta_spec(with_L)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    fracs = np.linspace(0.25, 0.75, n_starts)
    best = None
    for fr in fracs:
        x0 = lo + fr * (hi - lo)
        res = minimize(_nmll, x0, args=(keys, A_train, y, log_tau, psi_train, with_L),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    return dict(zip(keys, best.x)), float(best.fun)


# Gaussian posterior of the latent
def posterior(theta: dict, A_obs: np.ndarray, y: np.ndarray, Gamma_all: np.ndarray):
    """Gaussian posterior of the latent: mean mu_x = Gamma A^T Psi^-1 y and covariance
    Sig_x = Gamma - Gamma A^T Psi^-1 A Gamma, with Psi = A Gamma A^T + sigma_n^2 I and A = A_obs
    (dense Cholesky solve).
    """
    n = A_obs.shape[0]
    Xi = A_obs @ Gamma_all                         # (n_obs, D)
    Psi = Xi @ A_obs.T + (theta["sigma_n"] ** 2) * np.eye(n)
    cf = _chol(Psi)
    mu_x = Xi.T @ cho_solve(cf, y, check_finite=False)
    Sig_x = Gamma_all - Xi.T @ cho_solve(cf, Xi, check_finite=False)
    return mu_x, Sig_x


# fixed HMC seed of each held-out state, keyed by (dataset, v_nom). The column `seed` of
# experiments/02_qgpdrt.csv records the seed of each fit.
HMC_SEEDS = {
    ("D1_zhang", 3.0): 3639310351,
    ("D1_zhang", 3.05): 1199402735,
    ("D1_zhang", 3.4): 995735120,
    ("D1_zhang", 3.7): 4144133851,
    ("D1_zhang", 4.0): 2624252593,
    ("D1_zhang", 4.2): 1961382779,
    ("D2_rashid", 3.3): 3707248279,
    ("D2_rashid", 3.59): 1165077836,
    ("D2_rashid", 3.69): 1290381145,
    ("D2_rashid", 3.85): 998529112,
    ("D2_rashid", 4.15): 1915568009,
    ("D3_mustafa", 3.1715): 781826852,
    ("D3_mustafa", 3.206): 617870717,
    ("D3_mustafa", 3.2261): 2335177874,
    ("D3_mustafa", 3.2491): 2042579994,
    ("D3_mustafa", 3.2631): 2768466450,
    ("D3_mustafa", 3.278): 3594882041,
    ("D3_mustafa", 3.2842): 264069110,
    ("D3_mustafa", 3.2867): 3965105953,
    ("D3_mustafa", 3.2875): 930586175,
    ("D3_mustafa", 3.2887): 2057750255,
    ("D3_mustafa", 3.2908): 973135323,
    ("D3_mustafa", 3.2996): 3908087933,
    ("D3_mustafa", 3.3139): 3051660257,
    ("D3_mustafa", 3.3248): 2096954420,
    ("D3_mustafa", 3.3264): 706828866,
    ("D3_mustafa", 3.3269): 647843125,
    ("D3_mustafa", 3.327): 3046963364,
    ("D3_mustafa", 3.3274): 1568222972,
    ("D3_mustafa", 3.3284): 132623542,
    ("D3_mustafa", 3.4377): 4176905830,
}


def fold_seed(dataset: str, held_out_v_nom: float) -> int:
    """HMC seed of the held-out state `held_out_v_nom` of `dataset` from `HMC_SEEDS`.

    Raises KeyError for a state outside the table. A seed can also go to `predict_loso` directly.
    """
    key = (dataset, float(held_out_v_nom))
    if key not in HMC_SEEDS:
        raise KeyError(f"no HMC seed for dataset {dataset!r}, state v_nom={held_out_v_nom!r}; "
                       f"pass seed= to predict_loso")
    return HMC_SEEDS[key]


# HMC over the latent block of the held-out state
def block_constraint_F(bd, with_L):
    """Constraint F x + g >= 0 of one latent block: R >= 0 and gamma >= 0. L is free."""
    g0 = gamma_start(with_L)
    rows = [0] + list(range(g0, bd))                             # R, gamma(1..T)
    F = np.zeros((len(rows), bd))
    F[np.arange(len(rows)), rows] = 1.0
    return F, np.zeros(len(rows))


def sample_block_truncated(mu_b, Sig_b, bd, with_L, seed, L=5000):
    """L HMC samples (vendored `generate_tmg`) of the latent block of the held-out state.

    The target is the Gaussian marginal N(mu_b, Sig_b) of this block, truncated to R >= 0 and
    gamma >= 0 with L free. The blocks of the training states are not sampled. The global numpy
    random state is seeded with `seed` before the call. Returns (samples of shape (bd, L),
    number of rejected HMC proposals).
    """
    g0 = gamma_start(with_L)
    F, g = block_constraint_F(bd, with_L)
    init = np.array(mu_b, dtype=float, copy=True)
    init[0] = max(abs(init[0]), 1e-6)                            # R
    init[g0:] = np.maximum(np.abs(init[g0:]), 1e-6)              # gamma
    if np.any(F @ init + g < 0):
        raise AssertionError("infeasible HMC init (F x + g < 0)")
    M = Sig_b + 1e-8 * np.eye(bd)
    np.random.seed(seed)
    buf = io.StringIO()
    with redirect_stdout(buf):
        samples = _vhmc.generate_tmg(F, g, M, mu_b, init, cov=True, L=L)
    n_reject = buf.getvalue().count("hmc reject")
    return samples, n_reject


# prediction of one held-out state
def predict_loso(dataset, held_out_v_nom, freq, psi_raw_train, psi_raw_star,
                 re_train, im_train, T, with_L, L=5000, burn=1000, seed=None):
    """Predict the spectrum of the held-out state from the training states.

    freq                : common frequency grid (F,), ascending, in Hz.
    psi_raw_train       : (S-1,) state coordinates v of the training states.
    psi_raw_star        : state coordinate v of the held-out state.
    re_train, im_train  : (S-1, F) Re Z and Im Z of the training states, in ohm.
    T                   : number of tau values.
    L, burn             : HMC samples and the number dropped at the start (at most L // 5).
    seed                : HMC seed (None takes `fold_seed(dataset, held_out_v_nom)`).

    The measured values of the held-out state are not an argument. Returns mu_re, mu_im (mean of
    the HMC predictive), sd_re, sd_im (its standard deviation with the noise variance added),
    re_lo, re_hi, im_lo, im_hi (2.5 and 97.5 percentiles of the samples plus noise), all at freq
    in ohm, and the fitted quantities (theta, nmll, c, seed, n_reject and others).
    """
    psi_raw_train = np.asarray(psi_raw_train, dtype=float)
    re_train = np.asarray(re_train, dtype=float)             # (S-1, F)
    im_train = np.asarray(im_train, dtype=float)
    S_train = psi_raw_train.size
    F = freq.size

    # standardization with the training states only
    mu_psi = float(psi_raw_train.mean())
    sd_psi = float(psi_raw_train.std())
    if not np.isfinite(sd_psi) or sd_psi == 0.0:               # then scale by the range
        rng = float(psi_raw_train.max() - psi_raw_train.min())
        sd_psi = rng if rng > 0 else 1.0
    psi_train = (psi_raw_train - mu_psi) / sd_psi
    psi_star = (float(psi_raw_star) - mu_psi) / sd_psi
    c = float(np.std(np.concatenate([re_train.ravel(), im_train.ravel()])))
    if not np.isfinite(c) or c == 0.0:
        c = 1.0

    # tau grid, which must cover 1 / (2 pi f) of every frequency
    tau, log_tau = build_tau_grid(freq, T)
    inv = 1.0 / (TWO_PI * freq)
    assert tau.min() <= inv.min() + 1e-12 and tau.max() >= inv.max() - 1e-12, \
        "tau grid does not cover all observed 1/(2 pi f)"
    bd = block_dim(T, with_L)

    # operators
    A_state = state_A(freq, tau, with_L)                       # (2F, bd)
    A_train = blockdiag(A_state, S_train)                      # (2F(S-1), bd(S-1))

    # training observations / c, state-major [re_s; im_s]
    y = np.concatenate([np.concatenate([re_train[s], im_train[s]]) for s in range(S_train)]) / c

    # hyperparameters from the training states only
    theta, nmll_val = fit_theta(A_train, y, log_tau, psi_train, with_L)

    # latent of the training states and the held-out state (appended last)
    psi_all = np.concatenate([psi_train, [psi_star]])
    S_all = psi_all.size                                       # S_train + 1
    Gamma_all = assemble_Gamma(theta, log_tau, psi_all, with_L)
    A_obs = np.zeros((2 * F * S_train, bd * S_all))
    for k in range(S_train):
        A_obs[2 * F * k:2 * F * (k + 1), bd * k:bd * (k + 1)] = A_state
    # the columns of the held-out block (index S_train) stay zero (no observation rows)

    mu_x, Sig_x = posterior(theta, A_obs, y, Gamma_all)

    # HMC over the held-out block only (sample_block_truncated)
    if seed is None:
        seed = fold_seed(dataset, held_out_v_nom)
    hb = slice(bd * S_train, bd * S_all)                       # held-out latent block
    mu_b = mu_x[hb]
    Sig_b = Sig_x[hb, hb]
    samples, n_reject = sample_block_truncated(mu_b, Sig_b, bd, with_L, seed, L=L)

    # predictive of the held-out state: A_s applied to the samples
    burn = min(burn, L // 5)                                   # drop at most a fifth
    x_star = samples[:, burn:]                                 # (bd, L-burn)
    Z_std = A_state @ x_star                                   # (2F, L-burn), standardized
    re_std = Z_std[:F, :]
    im_std = Z_std[F:, :]
    sigma_n = theta["sigma_n"]

    mu_re = c * re_std.mean(axis=1)
    mu_im = c * im_std.mean(axis=1)
    sd_re = c * np.sqrt(re_std.var(axis=1) + sigma_n ** 2)     # plus the noise variance
    sd_im = c * np.sqrt(im_std.var(axis=1) + sigma_n ** 2)

    # interval of an observation: samples plus noise from a generator seeded with `seed`
    noise_rng = np.random.default_rng(seed)
    re_obs = c * (re_std + sigma_n * noise_rng.standard_normal(re_std.shape))
    im_obs = c * (im_std + sigma_n * noise_rng.standard_normal(im_std.shape))
    re_lo, re_hi = np.percentile(re_obs, [2.5, 97.5], axis=1)
    im_lo, im_hi = np.percentile(im_obs, [2.5, 97.5], axis=1)

    return {
        "mu_re": mu_re, "mu_im": mu_im, "sd_re": sd_re, "sd_im": sd_im,
        "re_lo": re_lo, "re_hi": re_hi, "im_lo": im_lo, "im_hi": im_hi,
        "theta": theta, "nmll": nmll_val, "c": c, "mu_psi": mu_psi, "sd_psi": sd_psi,
        "psi_star_std": psi_star, "T": T, "with_L": with_L, "n_reject": int(n_reject),
        "seed": int(seed), "mu_x": mu_x, "Sig_x": Sig_x, "A_state": A_state,
        "Gamma_all": Gamma_all, "A_obs": A_obs, "y": y, "psi_all": psi_all,
        "log_tau": log_tau, "tau": tau,
    }


# the model with the inductance term (with_L=True)
VARIANTS = {"qGP-DRT": True}


def choose_T(dataset: str, freq: np.ndarray) -> int:
    """Number of tau values T = round(10 x the number of frequency decades).

    T is at most 30 on D3 and at most 80 on D1 and D2.
    """
    decades = float(np.log10(freq.max() / freq.min()))
    cap = 30 if dataset.startswith("D3") else 80
    return int(min(round(10 * decades), cap))


def fit_qgpdrt(dname: str, held_out: float, variant: str, L: int):
    """Predict the spectrum of the state `held_out` from the other states and score it."""
    with_L = VARIANTS[variant]
    df = data.load_dataset(dname)
    grid = build_grid(df)
    vnom = grid["v_nom"]
    omega = grid["omega"]
    v = grid["v"]
    states = sorted(set(vnom))
    v_by_state = {s: float(v[vnom == s][0]) for s in states}
    v_min, v_max = min(v_by_state.values()), max(v_by_state.values())

    mask = vnom != held_out                                   # training rows
    if mask[vnom == held_out].any():
        raise AssertionError(f"LEAK: held-out state {held_out} in train ({dname})")
    if not mask[vnom != held_out].all():
        raise AssertionError(f"LEAK: a training state dropped ({dname},{held_out})")

    held_rows = np.where(vnom == held_out)[0]
    freq = omega[held_rows] / TWO_PI                          # common grid, f ascending
    train_states = [s for s in states if s != held_out]
    re_train, im_train, psi_train = [], [], []
    for s in train_states:
        rs = np.where(vnom == s)[0]
        if not np.allclose(omega[rs] / TWO_PI, freq):
            raise AssertionError(f"non-common frequency grid ({dname}, state {s})")
        re_train.append(grid["obs_re"][rs])
        im_train.append(grid["obs_im"][rs])
        psi_train.append(float(v[rs][0]))
    psi_star = float(v[held_rows][0])

    T = choose_T(dname, freq)
    t0 = time.perf_counter()
    pred = predict_loso(dname, float(held_out), freq, psi_train, psi_star,
                           re_train, im_train, T, with_L, L=L)
    fit_time = time.perf_counter() - t0

    pred_m = {"mu_re": pred["mu_re"], "mu_im": pred["mu_im"],
              "var_re": pred["sd_re"] ** 2, "var_im": pred["sd_im"] ** 2}
    m = case_metrics(grid, mask, pred_m)

    yr = grid["obs_re"][held_rows]
    yi = grid["obs_im"][held_rows]
    hmc_re = float(np.mean((yr >= pred["re_lo"]) & (yr <= pred["re_hi"])))
    hmc_im = float(np.mean((yi >= pred["im_lo"]) & (yi <= pred["im_hi"])))
    inside = np.concatenate([(yr >= pred["re_lo"]) & (yr <= pred["re_hi"]),
                             (yi >= pred["im_lo"]) & (yi <= pred["im_hi"])])
    hmc_all = float(np.mean(inside))

    v_s = v_by_state[held_out]
    region = "boundary" if (v_s == v_min or v_s == v_max) else "interior"
    th = pred["theta"]
    return {
        "dataset": dname, "method": variant, "circuit": "qGP-DRT", "with_L": with_L,
        "split": "cross_state", "held_out_v_nom": float(held_out), "held_out_v": v_s,
        "region": region, "n_train_states": len(train_states), **m,
        "hmc_cov95_re": hmc_re, "hmc_cov95_im": hmc_im, "hmc_cov95_all": hmc_all,
        "T": T, "n_reject": pred["n_reject"], "seed": pred["seed"],
        "sigma_n": th["sigma_n"], "nmll": pred["nmll"], "imp_scale_c": pred["c"],
        "fit_time_s": fit_time,
    }


def build_cases(dnames):
    cases = []
    for dname in dnames:
        datasets.resolve(dname)
        df = data.load_dataset(dname)
        states = sorted(df["v_nom"].unique())
        for s in states:
            for variant in VARIANTS:
                cases.append((dname, float(s), variant))
    return cases
