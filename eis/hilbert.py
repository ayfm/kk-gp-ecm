"""The discrete Kramers-Kronig operator H: Im Z ~= H @ Re Z on a fixed frequency grid.

The subtracted Kramers-Kronig relation:

    Im Z(omega) = (2 omega / pi) * P.V. integral_0^inf
                    [Re Z(s) - Re Z(omega)] / (s^2 - omega^2) ds.

On a grid omega_1 < ... < omega_n, the trapezoidal rule in u = log(omega), with the half weights
at the two ends, gives

    H0_ij = (2 omega_i / pi) * w_j / (omega_j^2 - omega_i^2)   (i != j),
    H0_ii = - sum_{k != i} H0_ik,

with w_j = omega_j * Delta u_j. Row i is the output frequency omega_i and column j the node
omega_j. The diagonal rule gives (H0 c)_i = sum_{j != i} H0_ij (c_j - c_i) and H0 @ 1 = 0.

With `order=1` the function `hilbert_matrix` returns H0. With `order=2` (the default) it returns
H = H0 + (1/pi) diag(Delta u) D, where D is a second-order finite-difference matrix of
d/d(log omega). This adds the node j = i, at which the subtracted integrand equals
Re'(omega_i) / (2 omega_i). Since D @ 1 = 0, also H @ 1 = 0.

The overall sign of H depends on the sign convention of the imaginary part. The function
`calibrate_sign` fixes it at import, so that H reproduces the analytic Im of
Z = R0 + R / (1 + R Q (j omega)^alpha) (a resistor in series with one ZARC, built with
`eis.ecm`). The result is the module constant `SIGN`, which `hilbert_matrix` applies by default.
"""

import numpy as np
import torch

from . import env, ecm

logger = env.logger


# the operator

def log_trapezoid_weights(omega: np.ndarray) -> np.ndarray:
    """Trapezoidal weights Delta u_j in u = log(omega), with half weights at the two ends.

    An interior node has the weight (u_{j+1} - u_{j-1}) / 2, and an end node half of the adjacent
    interval. The grid does not need to be uniform in u.
    """
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1 or omega.size < 2:
        raise ValueError("omega must be a 1-D array with at least 2 points.")
    if np.any(np.diff(omega) <= 0):
        raise ValueError("omega must be strictly increasing.")
    u = np.log(omega)
    dW = np.empty_like(u)
    dW[1:-1] = (u[2:] - u[:-2]) / 2.0
    dW[0] = (u[1] - u[0]) / 2.0
    dW[-1] = (u[-1] - u[-2]) / 2.0
    return dW


def _logderiv_matrix(omega: np.ndarray) -> np.ndarray:
    """Second-order finite-difference matrix D of d/d(log omega).

    Row i holds the three-point central difference at u_i = log(omega_i) on a non-uniform grid,
    with second-order one-sided stencils at the two ends, so (D @ c)_i ~= dc/du at node i.
    D @ 1 = 0.
    """
    u = np.log(np.asarray(omega, dtype=float))
    n = u.size
    if n < 3:
        raise ValueError("order=2 needs at least 3 grid points.")
    D = np.zeros((n, n))
    hm = u[1:-1] - u[:-2]                            # left spacings  (interior)
    hp = u[2:] - u[1:-1]                             # right spacings (interior)
    idx = np.arange(1, n - 1)
    D[idx, idx - 1] = -hp / (hm * (hm + hp))
    D[idx, idx] = (hp - hm) / (hm * hp)
    D[idx, idx + 1] = hm / (hp * (hm + hp))
    h0, h1 = u[1] - u[0], u[2] - u[1]                # left endpoint (forward)
    D[0, 0] = -(2 * h0 + h1) / (h0 * (h0 + h1))
    D[0, 1] = (h0 + h1) / (h0 * h1)
    D[0, 2] = -h0 / (h1 * (h0 + h1))
    g0, g1 = u[-1] - u[-2], u[-2] - u[-3]            # right endpoint (backward)
    D[-1, -1] = (2 * g0 + g1) / (g0 * (g0 + g1))
    D[-1, -2] = -(g0 + g1) / (g0 * g1)
    D[-1, -3] = g0 / (g1 * (g0 + g1))
    return D


def hilbert_matrix(omega: np.ndarray, sign: float | None = None,
                   order: int = 2) -> np.ndarray:
    """Discrete Kramers-Kronig operator H on the grid `omega` (Im ~= H @ Re).

    Parameters
    ----------
    omega : array-like (n,)
        Strictly increasing angular-frequency grid (rad/s).
    sign : {+1, -1}, optional
        Sign multiplier. Default: the calibrated module constant `SIGN`. +1 gives the operator
        of the quadrature formula itself.
    order : {1, 2}
        1 returns H0. 2 (default) returns H = H0 + (1/pi) diag(Delta u) D.

    Returns
    -------
    H : (n, n) float64 ndarray with H @ 1 = 0. H is not symmetric.
    """
    if sign is None:
        sign = SIGN
    omega = np.asarray(omega, dtype=float)
    dW = log_trapezoid_weights(omega)
    w = omega * dW                                  # w_j  (length n)

    wi = omega[:, None]                             # omega_i  (varies down rows = output i)
    wj = omega[None, :]                             # omega_j  (varies across cols = node j)
    denom = wj**2 - wi**2
    np.fill_diagonal(denom, 1.0)                    # avoids 0/0 (the diagonal is overwritten below)
    M = (2.0 * wi / np.pi) * w[None, :] / denom     # off-diagonal entries of H0
    np.fill_diagonal(M, 0.0)

    H = M.copy()
    np.fill_diagonal(H, -M.sum(axis=1))             # row-sum-zero rule of H0
    if order == 2:
        # order 2: add (1/pi) diag(Delta u) D. H @ 1 = 0 stays because D @ 1 = 0
        H = H + np.diag(dW / np.pi) @ _logderiv_matrix(omega)
    elif order != 1:
        raise ValueError("order must be 1 or 2.")
    return sign * H


def hilbert_blockdiag(omega: np.ndarray, state_id, sign: float | None = None,
                      order: int = 2) -> np.ndarray:
    """Block-diagonal operator H_G = I kron H that acts within each state.

    For each state s, H is built on the omega-sorted points of s and placed on the rows and
    columns of these points. Different states are never coupled. The points can come in any
    order, and the states do not need to share a grid.

    Parameters
    ----------
    omega : (N,) array-like
        Angular frequencies of all N points.
    state_id : (N,) array-like
        State label of each point.
    sign : {+1, -1}, optional
        Sign multiplier (default `SIGN`).
    order : {1, 2}
        As in `hilbert_matrix`.

    Returns
    -------
    (N, N) float64 ndarray with H_G @ 1 = 0, block-diagonal up to a permutation of the points.
    """
    omega = np.asarray(omega, dtype=float)
    state_id = np.asarray(state_id)
    if omega.shape != state_id.shape:
        raise ValueError("omega and state_id must have the same shape.")
    N = omega.size
    Hf = np.zeros((N, N))
    for s in np.unique(state_id):
        idx = np.where(state_id == s)[0]
        sidx = idx[np.argsort(omega[idx])]           # global indices, omega-sorted
        Hs = hilbert_matrix(omega[sidx], sign=sign, order=order)
        Hf[np.ix_(sidx, sidx)] = Hs
    return Hf


# sign calibration

def _zarc_test_signal(
    omega: np.ndarray,
    R0: float = 1.0,
    R: float = 1.0,
    Q: float = 1.0,
    alpha: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic (Re, Im) of Z = R0 + R / (1 + R Q (j omega)^alpha) on `omega`, from eis.ecm.

    R0 adds a constant to the real part, which H maps to zero (H @ 1 = 0).
    """
    omega_t = torch.as_tensor(np.asarray(omega, dtype=float), dtype=torch.float64)
    with torch.no_grad():
        z = ecm.Resistor(R0)(omega_t) + ecm.Zarc(R, Q, alpha)(omega_t)
    z = z.numpy()
    return z.real.copy(), z.imag.copy()


def _rel_err(pred: np.ndarray, target: np.ndarray, interior: int = 0) -> float:
    """Relative L2 error ||pred - target|| / ||target||, optionally without `interior` points
    at each end."""
    if interior > 0:
        s = slice(interior, -interior)
        pred, target = pred[s], target[s]
    denom = np.linalg.norm(target)
    return float(np.linalg.norm(pred - target) / denom) if denom > 0 else float("nan")


def calibrate_sign(
    omega: np.ndarray | None = None,
    *,
    interior: int = 5,
    zarc: dict | None = None,
    order: int = 2,
) -> dict:
    """Choose the sign of H that reproduces the analytic Im of a resistor plus one ZARC.

    Applies the operator with sign +1 to the analytic real part and compares +H@Re and -H@Re
    with the analytic imaginary part. The sign with the smaller relative error is returned.

    Parameters
    ----------
    omega : array-like, optional
        Grid. Default: 81 points log-spaced over [1e-4, 1e4] rad/s.
    interior : int
        Points left out at each end for the reported interior error.
    zarc : dict, optional
        Override of the test impedance {R0, R, Q, alpha}.
    order : {1, 2}
        As in `hilbert_matrix`.

    Returns
    -------
    dict with: sign (+1/-1), rel_err_full, rel_err_interior (at the chosen sign),
    rel_err_wrong_sign (full), n, omega_min, omega_max, zarc.
    """
    if omega is None:
        omega = np.logspace(-4, 4, 81)
    omega = np.asarray(omega, dtype=float)
    zp = {"R0": 1.0, "R": 1.0, "Q": 1.0, "alpha": 0.8, **(zarc or {})}

    re, im = _zarc_test_signal(omega, zp["R0"], zp["R"], zp["Q"], zp["alpha"])
    H_raw = hilbert_matrix(omega, sign=1.0, order=order)
    pred = H_raw @ re

    err_plus = _rel_err(pred, im)
    err_minus = _rel_err(-pred, im)
    sign = 1.0 if err_plus <= err_minus else -1.0

    err_full = min(err_plus, err_minus)
    err_wrong = max(err_plus, err_minus)
    err_interior = _rel_err(sign * pred, im, interior=interior)

    return {
        "sign": sign,
        "rel_err_full": err_full,
        "rel_err_interior": err_interior,
        "rel_err_wrong_sign": err_wrong,
        "n": int(omega.size),
        "omega_min": float(omega.min()),
        "omega_max": float(omega.max()),
        "zarc": zp,
    }


# sign of H, set once at import by the test above
SIGN: float = calibrate_sign()["sign"]
