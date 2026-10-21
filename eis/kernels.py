"""The RBF kernel of the latent real-channel residual."""

import numpy as np
import torch


def _as_f64(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def rbf_kernel(X1, X2, lengthscales, sigma_f=1.0) -> torch.Tensor:
    """RBF cross-kernel:

        k(x, x') = sigma_f^2 * exp( - sum_d (x_d - x'_d)^2 / (2 l_d^2) ).

    X1 has the shape (n1, d) and X2 the shape (n2, d), with one length scale l_d per input
    dimension. Returns (n1, n2). The arguments `lengthscales` and `sigma_f` may be tensors with
    requires_grad=True.
    """
    X1 = _as_f64(X1)
    X2 = _as_f64(X2)
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1, X2 must be 2-D (n, d).")
    ls = torch.as_tensor(lengthscales, dtype=torch.float64)
    d2 = torch.cdist(X1 / ls, X2 / ls) ** 2
    return (sigma_f ** 2) * torch.exp(-0.5 * d2)
