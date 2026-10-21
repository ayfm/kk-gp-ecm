"""The Kramers-Kronig-constrained GP and the unconstrained GP baseline.

In KKGP the latent real-channel residual u lives on the full grid G of every point, training and
test. The imaginary residual is H_G u, with the block-diagonal operator H_G of `eis.hilbert`, so
g = B u with B = [I ; H_G]. Let B_O be the rows of B at the training points (training real rows,
then training imaginary rows). The residual data r = (y - m)_O then has the marginal law

    r ~ N(0, M),    M = B_O K B_O^T + Sigma,    Sigma = blockdiag(sigma_re^2 I, sigma_im^2 I),

with the RBF Gram matrix K over the grid (eis.kernels.rbf_kernel). Inference is exact through the
Cholesky factor of M. The posterior mean of the latent residual is u* = K B_O^T M^-1 r, and the
prediction over the whole grid is m + B u*, so Im - m_im = H_G (Re - m_re) holds for the
prediction. The hyperparameters (sigma_f, l, sigma_re, sigma_im) maximize the log marginal
likelihood. The mean m is fixed: the circuit mean (label "M1") or zero (label "M2"). Only the
training rows enter the likelihood, and the test points enter through their frequencies.

GPRBF has two independent GPs, one per channel. Each has its own RBF kernel on (log omega, v), its
own noise and a zero mean ("zero", GP-RBF-0) or a fitted constant ("const", GP-RBF-c). The
channels are not coupled, and there is no circuit.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from . import env, hilbert, kernels

logger = env.logger

_LOG2PI = math.log(2.0 * math.pi)


class KKGP(nn.Module):
    """Exact Kramers-Kronig-constrained two-channel GP on the full grid.

    Parameters
    ----------
    kernel : {'rbf'}
        Base kernel on feat = (G, d) real inputs (log omega, v).
    mean : {'M1', 'M2'}
        Label of the mean: 'M1' for the circuit mean and 'M2' for the zero mean. In both cases
        the mean is the array passed to `set_data` as `mean_re`, `mean_im`.
    n_lengthscales : int
        Number of RBF length scales (one per input dimension).
    jitter : float
        Diagonal jitter added to the Gram matrix K.
    order : {1, 2}
        Quadrature order of the operator H (eis.hilbert.hilbert_matrix). Default 2.
    """

    def __init__(self, kernel: str = "rbf", mean: str = "M1",
                 n_lengthscales: int = 2, jitter: float = 1e-10, order: int = 2):
        super().__init__()
        if kernel != "rbf":
            raise ValueError(f"kernel must be 'rbf', got {kernel!r}")
        if mean not in ("M1", "M2"):
            raise ValueError(f"mean must be 'M1' or 'M2', got {mean!r}")
        self.kernel = kernel
        self.mean = mean
        self.jitter = jitter
        self.order = order
        n_ls = n_lengthscales

        # unconstrained hyperparameters (the positive values are their exponentials)
        self.log_sf = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_ls = nn.Parameter(torch.zeros(n_ls, dtype=torch.float64))
        self.log_noise_re = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
        self.log_noise_im = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))

        self._d = None   # filled by set_data

    # hyperparameters
    @property
    def sigma_f(self) -> torch.Tensor:
        return self.log_sf.exp()

    @property
    def lengthscales(self) -> torch.Tensor:
        return self.log_ls.exp()

    @property
    def sigma_re(self) -> torch.Tensor:
        return self.log_noise_re.exp()

    @property
    def sigma_im(self) -> torch.Tensor:
        return self.log_noise_im.exp()

    # kernel
    def _cross(self, feat1, feat2) -> torch.Tensor:
        return kernels.rbf_kernel(feat1, feat2, self.lengthscales, self.sigma_f)

    def _ft(self, feat) -> torch.Tensor:
        return torch.as_tensor(np.asarray(feat), dtype=torch.float64)

    # data
    def set_data(self, omega, state_id, feat, obs_re, obs_im, mean_re, mean_im, train_mask):
        """Register the full grid and the training mask.

        All arrays cover the full grid of G points. The mask `train_mask` (G,) is True at the
        training points, and only those enter the likelihood. The values obs_re and obs_im at the
        other points are never read.
        """
        omega = np.asarray(omega, dtype=float)
        state_id = np.asarray(state_id)
        train_mask = np.asarray(train_mask, dtype=bool)
        G = omega.size
        Hgg = hilbert.hilbert_blockdiag(omega, state_id, order=self.order)    # (G, G)
        BG = np.vstack([np.eye(G), Hgg])                                      # (2G, G)
        tr = np.where(train_mask)[0]
        te = np.where(~train_mask)[0]
        O = np.concatenate([tr, G + tr])                                      # train real, train imag
        self._d = {
            "G": G, "nT": tr.size, "tr": tr, "te": te,
            "feat": self._ft(feat),
            "obs_re": torch.as_tensor(np.asarray(obs_re), dtype=torch.float64),
            "obs_im": torch.as_tensor(np.asarray(obs_im), dtype=torch.float64),
            "mre": torch.as_tensor(np.asarray(mean_re), dtype=torch.float64),
            "mim": torch.as_tensor(np.asarray(mean_im), dtype=torch.float64),
            "Hgg": torch.as_tensor(Hgg, dtype=torch.float64),
            "BO": torch.as_tensor(BG[O], dtype=torch.float64),               # (2T, G)
            "O": O,
        }
        return self

    # linear algebra
    def _Kgg(self) -> torch.Tensor:
        d = self._d
        return self._cross(d["feat"], d["feat"]) + self.jitter * torch.eye(d["G"], dtype=torch.float64)

    def _r_obs(self) -> torch.Tensor:
        d = self._d; tr = d["tr"]
        rr = d["obs_re"][tr] - d["mre"][tr]
        ri = d["obs_im"][tr] - d["mim"][tr]
        return torch.cat([rr, ri])

    @staticmethod
    def _safe_cholesky(M: torch.Tensor) -> torch.Tensor:
        """Cholesky factor of M, or of M + jitter * I with the smallest jitter of 1e-8, 1e-7, ...,
        1e-1 that gives a factor.

        Raises the error of the first factorization if no jitter works (for example when M is not
        finite).
        """
        try:
            return torch.linalg.cholesky(M)
        except Exception:
            eye = torch.eye(M.shape[0], dtype=M.dtype)
            jit = 1e-8
            while jit < 1.0:
                try:
                    return torch.linalg.cholesky(M + jit * eye)
                except Exception:
                    jit *= 10.0
            raise

    def _chol(self):
        """Return (L, r_obs, Kgg, BO): the Cholesky factor of M, the residual, K and B_O."""
        d = self._d
        Kgg = self._Kgg()
        BO = d["BO"]
        nT = d["nT"]
        M = BO @ Kgg @ BO.T                                       # (2T, 2T)
        idx = torch.arange(nT)
        M[idx, idx] = M[idx, idx] + self.sigma_re ** 2            # real-channel noise
        M[idx + nT, idx + nT] = M[idx + nT, idx + nT] + self.sigma_im ** 2
        return self._safe_cholesky(M), self._r_obs(), Kgg, BO

    # marginal likelihood
    def neg_log_marginal_likelihood(self) -> torch.Tensor:
        if self._d is None:
            raise RuntimeError("call set_data(...) first.")
        L, r, _, _ = self._chol()
        alpha = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
        quad = r @ alpha
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        return 0.5 * (quad + logdet + r.numel() * _LOG2PI)

    def _clone_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def fit(self, adam_steps: int = 300, adam_lr: float = 0.05,
            lbfgs_outer: int = 5, lbfgs_max_iter: int = 100, log_every: int = 0) -> dict:
        """Fit the hyperparameters: Adam steps, then L-BFGS.

        The state after Adam is kept. After each L-BFGS call, the new state replaces it when its
        negative log marginal likelihood is finite and smaller. The kept state is restored at the
        end. An L-BFGS call that raises ends the L-BFGS part.
        """
        nll0 = float(self.neg_log_marginal_likelihood())
        if adam_steps > 0:
            opt = torch.optim.Adam(self.parameters(), lr=adam_lr)
            for _ in range(adam_steps):
                opt.zero_grad()
                loss = self.neg_log_marginal_likelihood()
                loss.backward()
                opt.step()

        best_nll = float(self.neg_log_marginal_likelihood())
        best_state = self._clone_state()

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
                logger.warning("KKGP LBFGS step aborted (%s); keeping best params.",
                               type(ex).__name__)
                break
            cur = float(self.neg_log_marginal_likelihood())
            if math.isfinite(cur) and cur < best_nll:
                best_nll = cur
                best_state = self._clone_state()
            if log_every and (k + 1) % log_every == 0:
                logger.info("  KKGP LBFGS %d/%d : nll=%.6e", k + 1, lbfgs_outer, cur)

        self.load_state_dict(best_state)
        nllf = float(self.neg_log_marginal_likelihood())
        return {"nll_init": nll0, "nll_final": nllf,
                "sigma_f": float(self.sigma_f),
                "lengthscales": self.lengthscales.detach().tolist(),
                "sigma_re": float(self.sigma_re), "sigma_im": float(self.sigma_im)}

    # posterior
    @torch.no_grad()
    def _posterior_real_latent(self):
        """Posterior of the latent residual u over the full grid.

        Returns (u_post, C_uu, Kgg): the mean (G,), the covariance (G, G) and the Gram matrix, with
        u_post = K B_O^T M^-1 r and C_uu = K - K B_O^T M^-1 B_O K.
        """
        L, r, Kgg, BO = self._chol()
        BK = BO @ Kgg                                            # (2T, G) = B_O K_GG
        alpha = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
        u_post = BK.T @ alpha                                    # (G,)
        C_uu = Kgg - BK.T @ torch.cholesky_solve(BK, L)          # (G, G)
        C_uu = 0.5 * (C_uu + C_uu.T)
        return u_post, C_uu, Kgg

    @torch.no_grad()
    def predict_full(self, observation_noise: bool = True) -> dict:
        """Prediction at all G grid points.

        Returns mu_re, mu_im, var_re, var_im over the full grid. The means satisfy
        mu_im - m_im = H_G (mu_re - m_re). With `observation_noise` the variances include the noise
        variance of each channel.
        """
        if self._d is None:
            raise RuntimeError("call set_data(...) first.")
        d = self._d
        Hgg = d["Hgg"]
        u_post, C_uu, _ = self._posterior_real_latent()

        mu_re = d["mre"] + u_post
        mu_im = d["mim"] + Hgg @ u_post

        var_re = torch.clamp(torch.diagonal(C_uu), min=0.0)
        HC = Hgg @ C_uu
        var_im = torch.clamp((HC * Hgg).sum(dim=1), min=0.0)     # diag(H C H^T)
        if observation_noise:
            var_re = var_re + self.sigma_re ** 2
            var_im = var_im + self.sigma_im ** 2

        return {"mu_re": mu_re.numpy(), "mu_im": mu_im.numpy(),
                "var_re": var_re.numpy(), "var_im": var_im.numpy(),
                "u_post": u_post.numpy(), "H_gg": Hgg.numpy()}

    @torch.no_grad()
    def predict(self, observation_noise: bool = True) -> dict:
        """Prediction at the test points (where train_mask is False).

        The key `te` gives their indices into the full grid.
        """
        full = self.predict_full(observation_noise=observation_noise)
        te = self._d["te"]
        return {"mu_re": full["mu_re"][te], "mu_im": full["mu_im"][te],
                "var_re": full["var_re"][te], "var_im": full["var_im"][te],
                "te": te.copy()}


class GPRBF(nn.Module):
    """Unconstrained baseline: one exact RBF GP per channel (Re, Im) on (log omega, v).

    There is no constraint block and no circuit, so `set_data` ignores `mean_re` and `mean_im`.
    With mean='zero' the mean is m = (0, 0) (GP-RBF-0). With mean='const' it is m = (c_re, c_im)
    with one fitted constant per channel (GP-RBF-c).

    Each channel c has its own sigma_f[c], length scales l[c] (one per input dimension) and noise
    sigma[c]. The hyperparameters of both channels minimize the sum of the two negative log marginal
    likelihoods. Only the training points of a channel enter its Gram matrix and likelihood, and the
    test inputs enter the cross-kernel only.

    The method `set_data` has the signature of `KKGP.set_data`, so the same calls feed both models.
    """

    def __init__(self, mean: str = "const", n_lengthscales: int = 2,
                 jitter: float = 1e-10):
        super().__init__()
        if mean not in ("zero", "const"):
            raise ValueError(f"mean must be 'zero' or 'const', got {mean!r}")
        self.mean = mean
        self.jitter = jitter
        # per-channel unconstrained hyperparameters (the positive values are their exponentials)
        self.log_sf = nn.Parameter(torch.zeros(2, dtype=torch.float64))        # [re, im]
        self.log_ls = nn.Parameter(torch.zeros(2, n_lengthscales, dtype=torch.float64))
        self.log_noise = nn.Parameter(torch.full((2,), -2.0, dtype=torch.float64))
        if mean == "const":
            self.c = nn.Parameter(torch.zeros(2, dtype=torch.float64))         # [re, im]
        self._d = None

    # data
    def set_data(self, omega, state_id, feat, obs_re, obs_im, mean_re, mean_im, train_mask):
        """Register the full grid and the training mask.

        The argument `feat` is the (G, d) real input matrix (log omega, v) over the full grid.
        The arguments `omega`, `state_id`, `mean_re` and `mean_im` are accepted for the signature
        of KKGP and ignored.
        """
        feat = np.asarray(feat, dtype=float)
        if feat.ndim != 2:
            raise ValueError("GPRBF requires feat = (G, d) real coords (log omega, v).")
        train_mask = np.asarray(train_mask, dtype=bool)
        tr = np.where(train_mask)[0]
        te = np.where(~train_mask)[0]
        self._d = {
            "G": feat.shape[0], "tr": tr, "te": te,
            "feat": torch.as_tensor(feat, dtype=torch.float64),
            "obs": [torch.as_tensor(np.array(obs_re, dtype=float), dtype=torch.float64),
                    torch.as_tensor(np.array(obs_im, dtype=float), dtype=torch.float64)],
        }
        return self

    # per-channel pieces
    def _channel_targets(self, ch: int) -> torch.Tensor:
        """Training residual of channel ch after the (zero or constant) mean."""
        d = self._d
        y = d["obs"][ch][d["tr"]]
        if self.mean == "const":
            y = y - self.c[ch]
        return y

    def _chol_train(self, ch: int):
        """Return (L, r): the Cholesky factor of the Gram matrix plus noise and the residual."""
        d = self._d
        Xtr = d["feat"][d["tr"]]
        sf = self.log_sf[ch].exp()
        ls = self.log_ls[ch].exp()
        noise = self.log_noise[ch].exp()
        K = kernels.rbf_kernel(Xtr, Xtr, ls, sf)
        n = K.shape[0]
        eye = torch.eye(n, dtype=torch.float64)
        K = K + (self.jitter + noise ** 2) * eye
        L = KKGP._safe_cholesky(K)
        return L, self._channel_targets(ch)

    # marginal likelihood
    def neg_log_marginal_likelihood(self) -> torch.Tensor:
        if self._d is None:
            raise RuntimeError("call set_data(...) first.")
        total = torch.zeros((), dtype=torch.float64)
        for ch in (0, 1):
            L, r = self._chol_train(ch)
            alpha = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
            quad = r @ alpha
            logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
            total = total + 0.5 * (quad + logdet + r.numel() * _LOG2PI)
        return total

    def _clone_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def fit(self, adam_steps: int = 300, adam_lr: float = 0.05,
            lbfgs_outer: int = 5, lbfgs_max_iter: int = 100, log_every: int = 0) -> dict:
        """Fit the hyperparameters: Adam steps, then L-BFGS, with the best state kept as in
        KKGP.fit."""
        nll0 = float(self.neg_log_marginal_likelihood())
        if adam_steps > 0:
            opt = torch.optim.Adam(self.parameters(), lr=adam_lr)
            for _ in range(adam_steps):
                opt.zero_grad()
                loss = self.neg_log_marginal_likelihood()
                loss.backward()
                opt.step()

        best_nll = float(self.neg_log_marginal_likelihood())
        best_state = self._clone_state()

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
                logger.warning("GPRBF LBFGS step aborted (%s); keeping best params.",
                               type(ex).__name__)
                break
            cur = float(self.neg_log_marginal_likelihood())
            if math.isfinite(cur) and cur < best_nll:
                best_nll = cur
                best_state = self._clone_state()
            if log_every and (k + 1) % log_every == 0:
                logger.info("  GPRBF LBFGS %d/%d : nll=%.6e", k + 1, lbfgs_outer, cur)

        self.load_state_dict(best_state)
        nllf = float(self.neg_log_marginal_likelihood())
        return {"nll_init": nll0, "nll_final": nllf,
                "sigma_f": self.log_sf.exp().detach().tolist(),
                "lengthscales": self.log_ls.exp().detach().tolist(),
                "noise": self.log_noise.exp().detach().tolist()}

    # posterior
    @torch.no_grad()
    def _predict_channel(self, ch: int, idx: np.ndarray, observation_noise: bool):
        """Posterior mean and variance of channel ch at grid rows `idx`."""
        d = self._d
        L, r = self._chol_train(ch)
        Xtr = d["feat"][d["tr"]]
        Xte = d["feat"][idx]
        sf = self.log_sf[ch].exp()
        ls = self.log_ls[ch].exp()
        noise = self.log_noise[ch].exp()
        Kst = kernels.rbf_kernel(Xte, Xtr, ls, sf)               # (n_idx, n_tr)
        alpha = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
        mu = Kst @ alpha
        if self.mean == "const":
            mu = mu + self.c[ch]
        # var = k(x,x) - q^T q with q = L^{-1} Kst^T and k(x,x) = sigma_f^2 for the RBF kernel
        Q = torch.linalg.solve_triangular(L, Kst.T, upper=False)  # (n_tr, n_idx)
        var = sf ** 2 - (Q * Q).sum(dim=0)
        var = torch.clamp(var, min=0.0)
        if observation_noise:
            var = var + noise ** 2
        return mu.numpy(), var.numpy()

    @torch.no_grad()
    def predict_full(self, observation_noise: bool = True) -> dict:
        """Prediction at all G grid points (both channels)."""
        if self._d is None:
            raise RuntimeError("call set_data(...) first.")
        allidx = np.arange(self._d["G"])
        mu_re, var_re = self._predict_channel(0, allidx, observation_noise)
        mu_im, var_im = self._predict_channel(1, allidx, observation_noise)
        return {"mu_re": mu_re, "mu_im": mu_im, "var_re": var_re, "var_im": var_im}

    @torch.no_grad()
    def predict(self, observation_noise: bool = True) -> dict:
        """Prediction at the test points (where train_mask is False)."""
        if self._d is None:
            raise RuntimeError("call set_data(...) first.")
        te = self._d["te"]
        mu_re, var_re = self._predict_channel(0, te, observation_noise)
        mu_im, var_im = self._predict_channel(1, te, observation_noise)
        return {"mu_re": mu_re, "mu_im": mu_im, "var_re": var_re, "var_im": var_im,
                "te": te.copy()}
