# Kramers-Kronig-Consistent Gaussian Process Prediction of Lithium-Ion Battery Impedance Spectra Across States of Charge
 

The impedance spectrum of a cell at a state $v$ is the complex function
$Z(\omega) = a(\omega) + j\,b(\omega)$ of the angular frequency $\omega = 2\pi f$. The code
predicts the spectrum of a held-out state from the spectra of the other states of the same
dataset. The predictor K-K GP-ECM is a Gaussian process (GP) whose mean is an equivalent
circuit $Z_\mathrm{ECM}(\omega; \theta)$. The correction of the GP to this mean satisfies the
discrete Kramers-Kronig (K-K) relation $b = Ha$. Here $H$ is a discrete Hilbert-transform matrix
on the frequency grid.

The repository holds the library `eis`, three public datasets and four notebooks. Each notebook
computes its results, saves them as files in `experiments/` and shows them as tables and figures.

## Layout

```
eis/                     library
  data.py, datasets.py     dataset names and loading of a workbook
  ecm.py, fitting.py       equivalent circuit, its fit to one spectrum, circuit mean of a held-out state
  hilbert.py               discrete Hilbert-transform matrix H and its block-diagonal form H_G
  kernels.py, gp.py        RBF kernel, K-K GP (KKGP) and unconstrained GP (GPRBF)
  gpexp.py, utils.py       grid, GP fit settings, metrics, output files of the notebooks
  loso.py                  one held-out state: GP-RBF-0, GP-RBF-c, K-K GP-0, K-K GP-ECM, ECM (circuit)
  qgpdrt.py                qGP-DRT under leave-one-state-out
  quasigp/                 unmodified files of the qGP-DRT code, with their license
  joint.py                 joint fit of the circuit and the GP
  consistency.py           sign errors, K-K residual, error floor, optimizer restarts
  linkk.py                 Lin-KK test of the measured spectra
  rawdata.py, build_d*.py  raw data download and check, workbook build and comparison
  plotting.py, env.py      figure style, paths and logging
datasets/                three workbooks, state labels of D1 and D2, MD5 list of the raw files
notebooks/               four notebooks
experiments/             output files of the notebooks
```

## Setup

The code runs on the CPU. The models of the library run in PyTorch. The baseline
qGP-DRT runs in NumPy and SciPy, and the Lin-KK test runs in the package `impedance`.
  

```bash
conda env create -f environment.yml
conda activate kk-gp-ecm
pip install -e .
```
 

```bash
python -c "from eis import data, loso; r = loso.fit_kkgp_zero('D2_rashid', 3.69, data.load_dataset('D2_rashid')); print(round(r['rmse_re'], 6), round(r['rmse_im'], 6))"
```
 

## Notebooks

Run the notebooks in the folder `notebooks/`, for example in JupyterLab. With
`RECOMPUTE = False` (the default) a notebook reads its output files in `experiments/` and
computes only a file that is missing. With `RECOMPUTE = True` it computes all its files again.
The setting `N_JOBS` is the number of worker processes of the fits. The notebooks are saved with
the outputs of a run that computed every file (`jupyter nbconvert --execute`, `N_JOBS = 3`, the
computer above).

| Notebook | Content | Output files |
|---|---|---|---|
| `01_datasets.ipynb` | datasets, measured spectra, Lin-KK test | `01_linkk_perstate.csv`, `01_linkk_residuals.csv`, `01_linkk_summary.csv` |
| `02_loso_prediction.ipynb` | leave-one-state-out prediction, RMSE by dataset and region | `02_kkgp_ecm.csv`, `02_gp_rbf.csv`, `02_kkgp_zero.csv`, `02_qgpdrt.csv` |
| `03_kk_consistency.ipynb` | sign errors, K-K residual, error floor of the constraint, optimizer restarts | `03_sign_errors.csv`, `03_rho_gp_rbf_c.csv`, `03_error_floor.csv`, `03_restarts.csv`, `03_signflip.npz` |
| `04_joint_fit.ipynb` | joint fit of the circuit and the GP against the two-step fit | `04_joint_fit.csv` |

The last column is the time of a run that computes every file, with `N_JOBS = 3` on the computer
above. Each of the 31 states (6 in D1, 5 in D2, 20 in D3) is held out once. The measured values
of the held-out state enter no fit. The GP and circuit fits use no random numbers. The HMC
sampler of qGP-DRT uses one fixed seed per held-out state (`HMC_SEEDS` in `eis/qgpdrt.py`).  

## References

| Item | Source | License |
|---|---|---|
| D1 (LCO/graphite) | Zhang et al., Zenodo, https://doi.org/10.5281/zenodo.3633835 |  
| D2 (NMC) | Rashid et al., Mendeley Data, https://doi.org/10.17632/mn9fb7xdx6.3 |  
| D3 (LFP) | Mustafa et al., Mendeley Data, https://doi.org/10.17632/cb887gkmxw.2 | 
| `eis/quasigp/` | https://github.com/ciuccislab/quasiGP (folder `tutorials/`) | 
| `impedance` (Lin-KK test) | Python package `impedance` 1.7.1 |
  
