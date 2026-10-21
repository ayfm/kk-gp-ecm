"""The joint fit of the circuit and the GP against the two-step fit, under leave-one-state-out.

The class `JointKKGPLoso` is `eis.gp.KKGP(kernel="rbf", mean="M1")` with one change in the mean.
The circuit mean of each training state is the output of its own circuit module, and the
parameters theta_s of these modules are free parameters of the negative log marginal likelihood,
together with the GP hyperparameters (sigma_f, l, sigma_re, sigma_im). The held-out state has no
data. Its mean is the circuit with parameters interpolated from the fitted training parameters
(`eis.ecm.interpolate_params`, as in `eis.fitting.ecm_mean_loso`), and it is set before each
prediction. The marginal covariance, the likelihood and the posterior are those of KKGP.

The likelihood uses M = B_tr K B_tr^T + Sigma_tr over the training rows. A training imaginary
row of B_tr is a row of H_G, which is block-diagonal per state, so it has nonzero entries only at
the frequencies of its own training state. The loss therefore depends neither on the values nor
on the frequencies of the held-out state. The held-out state enters the prediction only through
the cross-covariance and the interpolated mean.

For one held-out state, `fit_joint_and_two_step` fits K-K GP-ECM in two ways.
  - In the two-step fit the circuit is fitted to each training state by least squares
    (eis.ecm.fit_to_spectrum), and the held-out state receives the interpolated parameters. The
    GP hyperparameters then maximize the marginal likelihood with the circuit fixed (the fit of
    eis.loso.fit_kkgp_ecm).
  - In the joint fit the circuit parameters of the training states and the GP hyperparameters
    maximize one marginal likelihood (JointKKGPLoso). The fit starts at the least-squares
    parameters and the two-step hyperparameters and keeps its best point, so its training
    negative log marginal likelihood is never larger than that of the two-step fit (asserted
    per case). The held-out state receives the parameters interpolated from the jointly fitted
    ones.
The held-out values enter neither fit, and the function asserts the mask.

Its row holds the metric columns without prefix (rmse_*, cov95_*, width_*, nll_*) of the joint
fit and the same columns with the prefix plugin_ of the two-step fit. Further columns are
ls_ecm_rmse_re/im and joint_ecm_rmse_re/im (the interpolated circuit alone, with the
least-squares and with the joint parameters), plugin_gp_corr_* and joint_gp_corr_* (the largest
|GP prediction - circuit mean| over the held-out frequencies), plugin_mll and joint_mll (the final
negative log marginal likelihood of each fit), dmll = joint_mll - plugin_mll, dtheta_max,
dtheta_mean and dtheta_rms (change of the n_theta raw training parameters from their
least-squares values), the fit times and the hyperparameters.
"""

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import ecm, env, fitting, gp, utils
from .ecm import predict_z
from .gp import KKGP
from .gpexp import build_grid, case_metrics, FIT_KW

logger = env.logger

METHOD = "KKGP-ecmmean-joint"      # joint fit of K-K GP-ECM

JOINT_FIT_KW = dict(adam_steps=300, adam_lr=0.02, lbfgs_outer=8, lbfgs_max_iter=100)


class JointKKGPLoso(KKGP):
    """KKGP with a circuit mean (eis.ecm.ECMW) whose training parameters are fitted jointly.

    Usage:
      m = JointKKGPLoso(circuit_cls)
      m.set_data(omega, state_id, feat, obs_re, obs_im, mean_re0, mean_im0, mask)
      m.attach_ecms(omega, v_nom_full, v_full, train_states, modules, held_out_v_nom)
      m.fit_joint(...)
      pred = m.predict()

    The mask is v_nom != held_out_v_nom. The method `predict` first sets the mean of every row from
    the current circuit parameters (`refresh_frozen_mean`).
    """

    def __init__(self, circuit_cls, **kw):
        super().__init__(kernel="rbf", mean="M1", **kw)
        self.circuit_cls = circuit_cls
        self.ecms = nn.ModuleList()        # one circuit module per training state
        self.train_states = []             # v_nom of each training module
        self._train_rows = None            # full-grid rows of each training state
        self._train_v = None               # v of each training state (for the interpolation)
        self._omega_t = None
        self._held_rows = None             # full-grid rows of the held-out state
        self._held_v = None                # v of the held-out state

    # circuit modules of the training states
    def attach_ecms(self, omega, v_nom_full, v_full, train_states, modules, held_out_v_nom):
        """Register the circuit modules of the training states and the rows of the held-out state.

        The list `train_states` holds the v_nom of the training states in the order of `modules`.
        Raises ValueError unless the rows of the training states are the training points of
        `set_data` and the rows of the held-out state are all other rows of the grid.
        """
        if self._d is None:
            raise RuntimeError("call set_data(...) before attach_ecms(...).")
        v_nom_full = np.asarray(v_nom_full)
        v_full = np.asarray(v_full, dtype=float)
        self._omega_t = torch.as_tensor(np.asarray(omega, dtype=float), dtype=torch.float64)
        self.train_states = list(train_states)
        self.ecms = nn.ModuleList(modules)
        if len(self.ecms) != len(self.train_states):
            raise ValueError("modules and train_states length mismatch.")
        self._train_rows = [torch.as_tensor(np.where(v_nom_full == s)[0], dtype=torch.long)
                            for s in self.train_states]
        self._train_v = [float(v_full[np.where(v_nom_full == s)[0][0]]) for s in self.train_states]
        held_rows = np.where(v_nom_full == float(held_out_v_nom))[0]
        self._held_rows = torch.as_tensor(held_rows, dtype=torch.long)
        self._held_v = float(v_full[held_rows[0]])

        # the training rows are the training points, and the held-out rows are the rest
        train_rows_all = torch.cat(self._train_rows)
        G = self._d["G"]
        obs_rows = set(self._d["tr"].tolist())
        if set(train_rows_all.tolist()) != obs_rows:
            raise ValueError("training ECM rows != observed (masked) rows.")
        if set(held_rows.tolist()) != set(range(G)) - obs_rows:
            raise ValueError("held-out rows are not the exact complement of the training rows.")
        if held_rows.size == 0:
            raise ValueError("held-out state has no rows on the grid.")
        return self

    # mean of the training rows as a function of the circuit parameters
    def _live_mean_full(self):
        """(mre, mim) over the full grid as differentiable functions of the training parameters.

        The training rows hold the output of the circuit modules, and the held-out rows hold 0.
        Only the training rows enter `_r_obs`, so these zeros never enter the loss.
        """
        G = self._d["G"]
        mre = torch.zeros(G, dtype=torch.float64)
        mim = torch.zeros(G, dtype=torch.float64)
        for module, rows in zip(self.ecms, self._train_rows):
            z = module(self._omega_t[rows])
            mre = mre.index_add(0, rows, z.real)
            mim = mim.index_add(0, rows, z.imag)
        return mre, mim

    def _r_obs(self):
        """Training residual (y - m)_O with the mean of the circuit modules.

        Replaces KKGP._r_obs.
        """
        d = self._d
        tr = d["tr"]
        mre, mim = self._live_mean_full()
        rr = d["obs_re"][tr] - mre[tr]
        ri = d["obs_im"][tr] - mim[tr]
        return torch.cat([rr, ri])

    @torch.no_grad()
    def _held_out_mean(self):
        """Circuit at the held-out state with the current training parameters interpolated at its v
        (`eis.ecm.interpolate_params`). Returns (re, im) numpy arrays at its frequencies."""
        params_train = [m.params() for m in self.ecms]
        interp = ecm.interpolate_params(self.circuit_cls, np.asarray(self._train_v),
                                            params_train, self._held_v)
        held_model = self.circuit_cls(interp)
        z = predict_z(held_model, self._omega_t[self._held_rows].numpy())
        return z.real, z.imag

    @torch.no_grad()
    def refresh_frozen_mean(self):
        """Set the prediction mean d['mre'], d['mim']: the circuit modules at the training rows and
        the interpolated circuit at the held-out rows."""
        mre, mim = self._live_mean_full()
        mre = mre.clone(); mim = mim.clone()
        hre, him = self._held_out_mean()
        mre[self._held_rows] = torch.as_tensor(hre, dtype=torch.float64)
        mim[self._held_rows] = torch.as_tensor(him, dtype=torch.float64)
        self._d["mre"] = mre.detach().clone()
        self._d["mim"] = mim.detach().clone()
        return self

    @torch.no_grad()
    def predict_full(self, observation_noise: bool = True) -> dict:
        """KKGP.predict_full after `refresh_frozen_mean`."""
        self.refresh_frozen_mean()
        return super().predict_full(observation_noise=observation_noise)

    # joint fit
    def fit_joint(self, adam_steps: int = 300, adam_lr: float = 0.02,
                  lbfgs_outer: int = 8, lbfgs_max_iter: int = 100,
                  log_every: int = 0) -> dict:
        """Minimize the negative log marginal likelihood over the training circuit parameters and
        the GP hyperparameters together: Adam steps, then L-BFGS.

        The smallest finite value seen, starting with the value at the initial parameters, is
        kept and its parameters are restored at the end. The result therefore never has a larger
        negative log marginal likelihood than the start.
        """
        best_nll = float(self.neg_log_marginal_likelihood())
        nll0 = best_nll
        best_state = self._clone_state()

        if adam_steps > 0:
            opt = torch.optim.Adam(self.parameters(), lr=adam_lr)
            for _ in range(adam_steps):
                opt.zero_grad()
                loss = self.neg_log_marginal_likelihood()
                cur = float(loss)
                if math.isfinite(cur) and cur < best_nll:           # parameters that gave `loss`
                    best_nll = cur
                    best_state = self._clone_state()
                loss.backward()
                opt.step()

        lbfgs = torch.optim.LBFGS(self.parameters(), lr=0.5, max_iter=lbfgs_max_iter,
                                  history_size=20, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-10, tolerance_change=1e-14)

        def closure():
            lbfgs.zero_grad()
            loss = self.neg_log_marginal_likelihood()
            loss.backward()
            return loss

        for k in range(lbfgs_outer):
            try:
                lbfgs.step(closure)
            except Exception as ex:
                logger.warning("JointKKGPLoso LBFGS step aborted (%s); keeping best params.",
                               type(ex).__name__)
                break
            cur = float(self.neg_log_marginal_likelihood())
            if math.isfinite(cur) and cur < best_nll:
                best_nll = cur
                best_state = self._clone_state()
            if log_every and (k + 1) % log_every == 0:
                logger.info("  JointKKGPLoso LBFGS %d/%d : nll=%.6e", k + 1, lbfgs_outer, cur)

        self.load_state_dict(best_state)
        nllf = float(self.neg_log_marginal_likelihood())
        ls = self.lengthscales.detach().tolist()
        return {"nll_init": nll0, "nll_final": nllf,
                "sigma_f": float(self.sigma_f), "lengthscales": ls,
                "sigma_re": float(self.sigma_re), "sigma_im": float(self.sigma_im)}

    @torch.no_grad()
    def raw_theta_vec(self) -> torch.Tensor:
        """Raw (log or logit) parameters of all training circuit modules, in module order."""
        return torch.cat([p.detach().reshape(-1) for m in self.ecms for p in m.parameters()])


def ls_mean_models_loso(grid: dict, df: pd.DataFrame, held_out: float, cname: str, cls):
    """The circuit mean of `eis.fitting.ecm_mean_loso`, together with the fitted training models.

    Returns (mre, mim, train_states, train_v, train_models, v_s): the mean over the full grid, the
    v_nom and v of each training state with its least-squares circuit model, and the v of the
    held-out state.
    """
    G = grid["omega"].size
    mre = np.zeros(G); mim = np.zeros(G)
    train_states = []; train_v = []; train_models = []
    fits = {}
    for v_nom, g in df.groupby("v_nom", sort=True):
        if float(v_nom) == float(held_out):
            continue                                       # the held-out state is never fitted
        res = fitting.fit_state(cname, cls, g)
        vco = float(g["v"].iloc[0])
        fits[float(v_nom)] = {"v": vco, "params": res["params"]}
        rows = np.where(grid["v_nom"] == float(v_nom))[0]
        z = predict_z(res["model"], grid["omega"][rows])
        mre[rows] = z.real; mim[rows] = z.imag
        train_states.append(float(v_nom)); train_v.append(vco); train_models.append(res["model"])
    # held-out state: parameters interpolated across the training states
    v_train = np.array([fits[o]["v"] for o in fits])
    params_train = [fits[o]["params"] for o in fits]
    rows = np.where(grid["v_nom"] == float(held_out))[0]
    v_s = float(grid["v"][rows][0])
    held_model = cls(ecm.interpolate_params(cls, v_train, params_train, v_s))
    z = predict_z(held_model, grid["omega"][rows])
    mre[rows] = z.real; mim[rows] = z.imag
    return mre, mim, train_states, train_v, train_models, v_s


def _set_gp_hypers(dst, src) -> None:
    with torch.no_grad():
        dst.log_sf.copy_(src.log_sf)
        dst.log_ls.copy_(src.log_ls)
        dst.log_noise_re.copy_(src.log_noise_re)
        dst.log_noise_im.copy_(src.log_noise_im)


def fit_joint_and_two_step(dname: str, held_out: float, cname: str, df: pd.DataFrame):
    """Fit K-K GP-ECM in two ways with the state `held_out` withheld and score both there."""
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
    te = np.where(~mask)[0]
    mre, mim, train_states, train_v, ls_models, held_v = ls_mean_models_loso(
        grid, df, held_out, cname, cls)

    # two-step fit (the fit of eis.loso.fit_kkgp_ecm)
    plug = gp.KKGP(kernel="rbf", mean="M1")
    plug.set_data(grid["omega"], grid["state_id"], grid["feat"],
                  grid["obs_re"], grid["obs_im"], mre, mim, mask)
    t0 = time.perf_counter()
    plug_info = plug.fit(**FIT_KW)
    plug_time = time.perf_counter() - t0
    pred_plug = plug.predict(observation_noise=True)
    m_plug = case_metrics(grid, mask, pred_plug)

    # joint fit, started at the least-squares circuits and the two-step hyperparameters
    joint = JointKKGPLoso(cls)
    joint.set_data(grid["omega"], grid["state_id"], grid["feat"],
                   grid["obs_re"], grid["obs_im"], mre, mim, mask)
    modules = []
    for m in ls_models:
        mod = cls(); mod.load_state_dict(m.state_dict()); modules.append(mod)   # copy of the fit
    joint.attach_ecms(grid["omega"], grid["v_nom"], grid["v"], train_states, modules, held_out)
    _set_gp_hypers(joint, plug)
    raw_ls = torch.cat([p.detach().reshape(-1) for m in modules for p in m.parameters()]).clone()

    t0 = time.perf_counter()
    joint_info = joint.fit_joint(**JOINT_FIT_KW)
    joint_time = time.perf_counter() - t0
    pred_joint = joint.predict(observation_noise=True)
    m_joint = case_metrics(grid, mask, pred_joint)

    if joint_info["nll_final"] > plug_info["nll_final"] + 1e-6:
        raise AssertionError(f"joint MLL worse than plug-in ({dname},{held_out},{cname}): "
                             f"{joint_info['nll_final']:.6e} > {plug_info['nll_final']:.6e}")

    # comparison columns
    jmre = joint._d["mre"].numpy(); jmim = joint._d["mim"].numpy()   # mean set by predict()
    dtheta = (joint.raw_theta_vec() - raw_ls).abs()
    v_s = v_by_state[held_out]
    region = "boundary" if (v_s == v_min or v_s == v_max) else "interior"
    ls_p, jl_p = plug_info["lengthscales"], joint_info["lengthscales"]
    plug_cols = {f"plugin_{k}": v for k, v in m_plug.items() if k != "n_test"}
    return {
        "dataset": dname, "method": METHOD, "circuit": cname, "split": "cross_state",
        "held_out_v_nom": float(held_out), "held_out_v": v_s, "region": region,
        "n_train_states": len(states) - 1, **m_joint, **plug_cols,
        # the interpolated circuit alone at the held-out state, least-squares and joint parameters
        "ls_ecm_rmse_re": utils.rmse(grid["obs_re"][te], mre[te]),
        "ls_ecm_rmse_im": utils.rmse(grid["obs_im"][te], mim[te]),
        "joint_ecm_rmse_re": utils.rmse(grid["obs_re"][te], jmre[te]),
        "joint_ecm_rmse_im": utils.rmse(grid["obs_im"][te], jmim[te]),
        # largest GP correction to the circuit mean of each fit at the held-out frequencies
        "plugin_gp_corr_re": float(np.abs(pred_plug["mu_re"] - mre[te]).max()),
        "plugin_gp_corr_im": float(np.abs(pred_plug["mu_im"] - mim[te]).max()),
        "joint_gp_corr_re": float(np.abs(pred_joint["mu_re"] - jmre[te]).max()),
        "joint_gp_corr_im": float(np.abs(pred_joint["mu_im"] - jmim[te]).max()),
        "plugin_mll": plug_info["nll_final"], "joint_mll": joint_info["nll_final"],
        "dmll": joint_info["nll_final"] - plug_info["nll_final"],
        "dtheta_max": float(dtheta.max()), "dtheta_mean": float(dtheta.mean()),
        "dtheta_rms": float((dtheta ** 2).mean().sqrt()), "n_theta": int(dtheta.numel()),
        "plugin_fit_time_s": plug_time, "joint_fit_time_s": joint_time,
        "plugin_sigma_f": plug_info["sigma_f"], "joint_sigma_f": joint_info["sigma_f"],
        "plugin_ls_logw": ls_p[0], "plugin_ls_v": ls_p[1],
        "joint_ls_logw": jl_p[0], "joint_ls_v": jl_p[1],
        "plugin_noise_re": plug_info["sigma_re"], "plugin_noise_im": plug_info["sigma_im"],
        "joint_noise_re": joint_info["sigma_re"], "joint_noise_im": joint_info["sigma_im"],
    }
