# Source of the qGP-DRT Files

The files `qGP_DRT.py`, `HMC.py` and `nearest_PD.py` are copies of the files of the same names in
https://github.com/ciuccislab/quasiGP. They come from the folder `tutorials/` at commit
503e0fdd6cd332ef9ed646ad9ef1b90caca28ab0. They are not modified. The file `LICENSE` is the MIT
license of that repository.

The method is described in A. Maradesa, B. Py and F. Ciucci, "Probabilistic deconvolution of the
distribution of relaxation times from multiple electrochemical impedance spectra", Journal of
Power Sources 621, 235236 (2024), https://doi.org/10.1016/j.jpowsour.2024.235236.

The module `eis/qgpdrt.py` uses `compute_A_re` and `compute_A_im` of `qGP_DRT.py` and
`generate_tmg` of `HMC.py`, which calls `nearest_PD.py`.
