"""The ten-parameter equivalent circuit ECM-W in PyTorch.

    Z_ECM(omega; theta) = j omega L + R_e + Z_Z(R_SEI, Q_SEI, alpha_SEI)
                          + Z_Z(R_ct, Q_dl, alpha_dl) + Z_W(A_w, tau),

with the ZARC Z_Z(R, Q, alpha) = R / (1 + R Q (j omega)^alpha) and the finite-length open
Warburg Z_W(A_w, tau) = (A_w / sqrt(j omega)) coth(sqrt(j omega tau)).

Each element is an `nn.Module`. Positive parameters (R, Q, A_w, L, tau) are stored as
raw = log(value), and the exponents alpha in (0, 1) as raw = logit(value). Every impedance is a
`torch.complex128` tensor. A circuit is fitted to one spectrum by least squares with
`fit_to_spectrum` (Adam steps, then L-BFGS). The parameters of a state without data come from
the fitted parameters of other states through `interpolate_params`.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from . import env

logger = env.logger


# parameter transforms

class _PositiveParam(nn.Module):
    """Stores `raw` unconstrained and exposes `value() = exp(raw) > 0`."""

    def __init__(self, init: float):
        super().__init__()
        if init <= 0:
            raise ValueError(f"_PositiveParam init must be > 0, got {init}")
        self.raw = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float64))

    def value(self) -> torch.Tensor:
        return self.raw.exp()


class _BoundedParam(nn.Module):
    """Stores `raw` unconstrained and exposes `value() = lo + (hi-lo)*sigmoid(raw)`."""

    def __init__(self, init: float, lo: float = 0.0, hi: float = 1.0):
        super().__init__()
        if not (lo < init < hi):
            raise ValueError(
                f"_BoundedParam init must be strictly inside ({lo}, {hi}), got {init}"
            )
        x = (init - lo) / (hi - lo)
        x = min(max(x, 1e-6), 1 - 1e-6)
        raw_init = math.log(x / (1.0 - x))
        self.raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float64))
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float64))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float64))

    def value(self) -> torch.Tensor:
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw)


# circuit elements

def _j_omega(omega: torch.Tensor) -> torch.Tensor:
    """Return j omega as a complex128 tensor."""
    return 1j * omega.to(torch.complex128)


class Resistor(nn.Module):
    """Z(omega) = R."""

    def __init__(self, R_init: float):
        super().__init__()
        self.R = _PositiveParam(R_init)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        return self.R.value().to(torch.complex128) * torch.ones_like(
            omega, dtype=torch.complex128
        )


class Inductor(nn.Module):
    """Z(omega) = j omega L."""

    def __init__(self, L_init: float):
        super().__init__()
        self.L = _PositiveParam(L_init)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        return _j_omega(omega) * self.L.value()


class Zarc(nn.Module):
    """Resistor in parallel with a constant-phase element (ZARC).

    Z = R / (1 + R Q (j omega)^alpha).
    """

    def __init__(self, R_init: float, Q_init: float, alpha_init: float):
        super().__init__()
        self.R = _PositiveParam(R_init)
        self.Q = _PositiveParam(Q_init)
        self.alpha = _BoundedParam(alpha_init, lo=0.0, hi=1.0)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        R = self.R.value()
        Q = self.Q.value()
        a = self.alpha.value()
        return R / (1.0 + R * Q * _j_omega(omega) ** a)


def _coth_stable(x: torch.Tensor) -> torch.Tensor:
    """Return coth(x) = (1 + exp(-2x)) / (1 - exp(-2x)).

    This form does not overflow for Re(x) > 0.
    """
    exp_neg_2x = torch.exp(-2.0 * x)
    return (1.0 + exp_neg_2x) / (1.0 - exp_neg_2x)


class WarburgOpen(nn.Module):
    """Finite-length open Warburg: Z(omega) = (A / sqrt(j omega)) coth(sqrt(j omega tau)).

    A is the amplitude and tau the diffusion time constant in seconds. For omega tau >> 1,
    coth -> 1 and Z -> A / sqrt(j omega). For omega tau << 1, Z -> A / (j omega sqrt(tau)).
    """

    def __init__(self, A_init: float = 1.0, tau_init: float = 10.0):
        super().__init__()
        self.A = _PositiveParam(A_init)
        self.tau = _PositiveParam(tau_init)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        j_om = _j_omega(omega)
        sqrt_j_om = j_om.sqrt()
        arg = (j_om * self.tau.value()).sqrt()
        z_w = self.A.value() / sqrt_j_om
        return z_w * _coth_stable(arg)


# the circuit

class _ECMBase(nn.Module):
    """Base class of a circuit with a `params()` dict and a string name."""

    NAME: str = "ECM"

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def params(self) -> dict[str, float]:
        raise NotImplementedError

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.4g}" for k, v in self.params().items())
        return f"{self.NAME}({items})"


class ECMW(_ECMBase):
    """The circuit ECM-W: L + R_e + Zarc(SEI) + Zarc(ct, dl) + WarburgOpen.

    Ten parameters: L, R_e, R_SEI, Q_SEI, alpha_SEI, R_ct, Q_dl, alpha_dl, A_w, tau.
    """

    NAME = "ECM-W"
    PARAM_KEYS = (
        "L", "R_e",
        "R_SEI", "Q_SEI", "alpha_SEI",
        "R_ct", "Q_dl", "alpha_dl",
        "A_w", "tau",
    )

    def __init__(self, init: Optional[dict] = None):
        super().__init__()
        d = init or {}
        self.ind = Inductor(d.get("L", 1e-6))
        self.re  = Resistor(d.get("R_e", 0.8))
        self.zarc_sei = Zarc(
            d.get("R_SEI", 0.1), d.get("Q_SEI", 1e-4), d.get("alpha_SEI", 0.8)
        )
        self.zarc_ct = Zarc(
            d.get("R_ct", 1.5), d.get("Q_dl", 1e-3), d.get("alpha_dl", 0.8)
        )
        self.warburg = WarburgOpen(d.get("A_w", 1.0), d.get("tau", 10.0))

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        return (
            self.ind(omega)
            + self.re(omega)
            + self.zarc_sei(omega)
            + self.zarc_ct(omega)
            + self.warburg(omega)
        )

    def params(self) -> dict[str, float]:
        return {
            "L":         float(self.ind.L.value()),
            "R_e":       float(self.re.R.value()),
            "R_SEI":     float(self.zarc_sei.R.value()),
            "Q_SEI":     float(self.zarc_sei.Q.value()),
            "alpha_SEI": float(self.zarc_sei.alpha.value()),
            "R_ct":      float(self.zarc_ct.R.value()),
            "Q_dl":      float(self.zarc_ct.Q.value()),
            "alpha_dl":  float(self.zarc_ct.alpha.value()),
            "A_w":       float(self.warburg.A.value()),
            "tau":       float(self.warburg.tau.value()),
        }


# fitting

def _to_torch(arr, dtype) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype)
    return torch.tensor(np.asarray(arr), dtype=dtype)


def _squared_loss(z_pred: torch.Tensor, z_meas: torch.Tensor) -> torch.Tensor:
    """Mean of (Re^2 + Im^2) of the complex residual."""
    r = z_pred - z_meas
    return (r.real ** 2 + r.imag ** 2).mean()


def fit_to_spectrum(
    ecm: _ECMBase,
    omega,
    z,
    *,
    adam_steps: int = 500,
    adam_lr: float = 0.05,
    lbfgs_max_iter: int = 200,
    lbfgs_outer: int = 5,
    lbfgs_lr: float = 0.5,
    history_size: int = 20,
    log_every: int = 0,
) -> dict:
    """Fit `ecm` in place to one spectrum: Adam steps, then L-BFGS with a strong-Wolfe line search.

    The loss is the mean of |Z_ECM(omega_i) - z_i|^2 over the points of the spectrum, which has
    the same minimizer as the sum of these squares.

    Parameters
    ----------
    ecm : _ECMBase
        Circuit instance. Its parameters are updated in place.
    omega : array-like (N,)
        Angular frequencies (rad/s).
    z : complex array-like (N,)
        Measured complex impedances.
    adam_steps : int
        Number of Adam steps (0 skips Adam).
    adam_lr : float
        Adam learning rate.
    lbfgs_max_iter : int
        L-BFGS iterations per call of `step`.
    lbfgs_outer : int
        Number of calls of the L-BFGS `step`.
    lbfgs_lr : float
        Initial L-BFGS step size.
    history_size : int
        L-BFGS history size.
    log_every : int
        Log the loss every `log_every` L-BFGS calls and once after Adam. 0 logs nothing.

    Returns
    -------
    dict with keys 'loss_init', 'loss_after_adam', 'loss_final', 'rmse', 'params'.
    """
    omega_t = _to_torch(omega, torch.float64)
    z_t = _to_torch(z, torch.complex128)

    with torch.no_grad():
        loss_init = float(_squared_loss(ecm(omega_t), z_t))

    # Adam
    loss_after_adam = loss_init
    if adam_steps > 0:
        adam = torch.optim.Adam(ecm.parameters(), lr=adam_lr)
        for _ in range(adam_steps):
            adam.zero_grad()
            loss = _squared_loss(ecm(omega_t), z_t)
            loss.backward()
            adam.step()
        with torch.no_grad():
            loss_after_adam = float(_squared_loss(ecm(omega_t), z_t))
        if log_every:
            logger.info(
                "  %s Adam %d steps : loss=%.4e -> %.4e",
                ecm.NAME, adam_steps, loss_init, loss_after_adam,
            )

    # L-BFGS
    lbfgs = torch.optim.LBFGS(
        ecm.parameters(),
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss = _squared_loss(ecm(omega_t), z_t)
        loss.backward()
        return loss

    for k in range(lbfgs_outer):
        lbfgs.step(closure)
        if log_every and (k + 1) % log_every == 0:
            with torch.no_grad():
                loss = float(_squared_loss(ecm(omega_t), z_t))
            logger.info(
                "  %s LBFGS %d/%d : loss=%.4e",
                ecm.NAME, k + 1, lbfgs_outer, loss,
            )

    with torch.no_grad():
        loss_final = float(_squared_loss(ecm(omega_t), z_t))

    return {
        "loss_init": loss_init,
        "loss_after_adam": loss_after_adam,
        "loss_final": loss_final,
        "rmse": math.sqrt(loss_final),
        "params": ecm.params(),
    }


def predict_z(ecm: _ECMBase, omega) -> np.ndarray:
    """Evaluate a fitted circuit on `omega` and return Z as a complex numpy array."""
    omega_t = _to_torch(omega, torch.float64)
    with torch.no_grad():
        z = ecm(omega_t)
    return z.cpu().numpy()


# data-driven initial parameters

def _spectrum_summary(omega: np.ndarray, z: np.ndarray, n_edge: int = 5) -> dict:
    """Summary values of a spectrum used as starting values of a circuit fit."""
    omega = np.asarray(omega, dtype=float)
    z = np.asarray(z, dtype=complex)
    order = np.argsort(omega)[::-1]  # highest frequency first
    omega = omega[order]
    z = z[order]

    R_hi = float(np.median(z.real[:n_edge]))             # high-frequency real
    R_lo = float(np.median(z.real[-n_edge:]))            # low-frequency real
    R_span = max(R_lo - R_hi, 0.05)

    # largest -Im(Z) without the two points at each end
    neg_im = -z.imag
    interior = neg_im[2:-2] if len(neg_im) > 4 else neg_im
    R_peak_height = float(max(interior.max(), 0.05))
    R_arc_total = 2.0 * R_peak_height                     # twice the peak height

    # A_w from the low-frequency tail: Re(Z) ~ const + (A_w / sqrt(2)) omega^(-1/2)
    tail_n = min(8, len(omega))
    omega_tail = omega[-tail_n:]
    re_tail = z.real[-tail_n:]
    x = 1.0 / np.sqrt(omega_tail)
    if np.ptp(x) > 0:
        slope = float(np.polyfit(x, re_tail, 1)[0])       # = A_w / sqrt(2)
        A_w_init = max(slope * math.sqrt(2.0), 0.05)
    else:
        A_w_init = 0.5

    return {
        "R_hi": R_hi,
        "R_lo": R_lo,
        "R_span": R_span,
        "R_arc": min(R_arc_total, R_span),
        "A_w": A_w_init,
    }


def init_from_spectrum(ecm_name: str, omega, z) -> dict:
    """Starting values of the circuit parameters from one spectrum.

    R_hi and R_lo are the medians of Re Z over the 5 highest and the 5 lowest frequencies, and
    P is the largest -Im Z without the two points at each end. Then
    R_arc = min(2 max(P, 0.05), max(R_lo - R_hi, 0.05)), and the starting values are:
    R_e = R_hi, R_SEI = max(0.3 R_arc, 0.05), R_ct = max(0.7 R_arc, 0.05), A_w = sqrt(2) times
    the slope of Re Z against omega^(-1/2) over the 8 lowest frequencies (at least 0.05, and 0.5
    if these frequencies are all equal),
    tau = 1 / max(omega_min, 1e-3), L = 1e-7, Q_SEI = 1e-4, Q_dl = 1e-3 and alpha = 0.8.
    """
    s = _spectrum_summary(np.asarray(omega), np.asarray(z))
    R_hi, R_arc, R_span, A_w = s["R_hi"], s["R_arc"], s["R_span"], s["A_w"]
    R_p1 = max(R_arc * 0.3, 0.05)
    R_p2 = max(R_arc * 0.7, 0.05)

    if ecm_name == "ECM-W":
        omega = np.asarray(omega, dtype=float)
        tau_init = float(1.0 / max(omega.min(), 1e-3))
        return {
            "L": 1e-7,
            "R_e": R_hi,
            "R_SEI": R_p1, "Q_SEI": 1e-4, "alpha_SEI": 0.8,
            "R_ct":  R_p2, "Q_dl":  1e-3, "alpha_dl":  0.8,
            "A_w": A_w,
            "tau": tau_init,
        }
    raise ValueError(f"Unknown ECM name: {ecm_name!r}")


# parameters across states
# A held-out state has no data. Its parameters come from the fitted parameters of the training
# states by linear interpolation in v, done in the space where the fit works: log for the
# positive parameters, logit for the alpha exponents.

def _is_alpha(key: str) -> bool:
    """True for the exponents alpha (they lie in (0, 1))."""
    return key.startswith("alpha")


def _to_unconstrained(key: str, x: float) -> float:
    """Map a parameter value to the space of the interpolation (log or logit)."""
    if _is_alpha(key):
        x = min(max(float(x), 1e-6), 1.0 - 1e-6)
        return math.log(x / (1.0 - x))
    return math.log(float(x))


def _from_unconstrained(key: str, y: float) -> float:
    """Inverse of `_to_unconstrained`."""
    if _is_alpha(key):
        return 1.0 / (1.0 + math.exp(-y))
    return math.exp(y)


def interpolate_params(
    ecm_class: type[_ECMBase],
    v_train: np.ndarray,
    params_train: list[dict],
    v_query: float,
) -> dict[str, float]:
    """Linear interpolation of each circuit parameter along the state coordinate v.

    Parameters
    ----------
    ecm_class : subclass of _ECMBase
        Defines `PARAM_KEYS` (the parameters to interpolate).
    v_train : array-like (S,)
        State coordinate of each training state.
    params_train : list of dict, length S
        `ecm.params()` of each training state, in the order of `v_train`.
    v_query : float
        State coordinate at which the parameters are wanted.

    Returns
    -------
    dict mapping each parameter name to its interpolated value.

    Notes
    -----
    The interpolation is linear in log (positive parameters) or logit (alpha). Outside
    [min v_train, max v_train], `np.interp` returns the value at the nearest end, so a boundary
    state receives the parameters of its nearest training state.
    """
    v_train = np.asarray(v_train, dtype=float)
    if len(v_train) != len(params_train):
        raise ValueError("v_train and params_train must have the same length.")
    order = np.argsort(v_train)
    v_sorted = v_train[order]
    out: dict[str, float] = {}
    for k in ecm_class.PARAM_KEYS:
        ys = np.array([_to_unconstrained(k, params_train[i][k]) for i in order])
        y_q = float(np.interp(float(v_query), v_sorted, ys))
        out[k] = _from_unconstrained(k, y_q)
    return out
