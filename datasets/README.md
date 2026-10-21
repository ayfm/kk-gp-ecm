# Datasets

The three workbooks in this folder are the data of the notebooks in `notebooks/`. Each
workbook holds one impedance spectrum per state of charge. A spectrum is the mean of the measured
complex impedance $Z(f) = \mathrm{Re}\,Z + j\,\mathrm{Im}\,Z$ over a group of cells at each
frequency $f$.
The library reads them with `eis.data.load_dataset("D1_zhang")` and the names `D2_rashid` and
`D3_mustafa`.

| File | States | Frequencies per state | Frequency range | MD5 |
|---|---|---|---|---|
| `D1_zhang_dataset.xlsx` | 6 | 60 | 0.02 Hz to 20 kHz | 7e068efcffb4f47ca09135a036532e1d |
| `D2_rashid_dataset.xlsx` | 5 | 61 | 0.01 Hz to 10 kHz | 5eb8519159b9b45c711f76b78b3c956e |
| `D3_mustafa_dataset.xlsx` | 20 | 28 | 0.01 Hz to 1 kHz | fc2f04be16bf16f3ae3a02be16db9613 |

## Layout of a Workbook

Each state has one sheet. The sheet name is the state label $v$ in volts.

| Column | Content |
|---|---|
| `f` | frequency in Hz |
| `zr` | $\mathrm{Re}\,Z$ in ohm |
| `zj` | $\mathrm{Im}\,Z$ in ohm |
| `zmod` | $\lvert Z\rvert$ in ohm |
| `zphz` | phase of $Z$ in degrees |
| `idc` | empty (not measured) |
| `vdc` | voltage in V. D1 and D2: the state label. D3: the mean measured voltage at that frequency. |

The sheet `meta` has one row per state. Its first columns are `v_nom` (the sheet name),
`v_actual` (the state label $v$), `n_points`, `f_min` and `f_max`. The column `state` (D1) or
`soc_pct` (D2, D3) holds the label of the state in the source data. The column
`n_cells_averaged` (D2) or `n_files_averaged` (D3) holds the number of averaged cells or files.

## Processing of the Raw Data

| Dataset | Raw files used | Processing |
|---|---|---|
| D1 | EIS files of cells 01 to 08 at 25 °C, cycle number at most 5, states I, II, III, IV, V and IX | Mean over cells and cycles at each frequency. The raw files store $-\mathrm{Im}\,Z$, so the sign is flipped. `zmod` and `zphz` are the means of the raw columns. The raw files have no voltage column, so each state has a nominal label: 3.05, 3.40, 3.70, 4.00, 4.20 and 3.00 V. |
| D2 | EIS files of cells 28 to 32 at 100% state of health and 25 °C, at 5, 20, 50, 70 and 95% state of charge | Mean over the 5 cells at each frequency. `zmod` and `zphz` are computed from the mean. The labels 3.30, 3.59, 3.69, 3.85 and 4.15 V come from a pseudo open-circuit-voltage model from the literature. They were not measured on these cells. |
| D3 | Hioki EIS files of the 11 cells, both test runs, state of charge 5% to 100% in steps of 5% | Mean over the 22 files at each frequency. `zmod` and `zphz` are computed from the mean. The label is the mean measured voltage. The sheet name rounds it to 4 decimals. |

The modules `eis.build_d1`, `eis.build_d2` and `eis.build_d3` hold this processing, and the
module `eis.rawdata` downloads and checks the raw data. With `DOWNLOAD_RAW = True` the notebook
`notebooks/01_datasets.ipynb` downloads the raw archives of D1, D2 and D3 into `datasets/raw/`.
It then checks the MD5 checksum of the 513 raw files that the build reads. Next it builds the
three workbooks into `datasets/rebuilt/` and compares them with the workbooks of this folder.

| File | Content |
|---|---|
| `D1_state_ocv_map.json`, `D2_soc_ocv_map.json` | state labels of D1 and D2 |
| `raw_files.md5` | MD5 checksums of the 513 raw files that the build reads |

The build of D2 reads the old binary Excel format of the D2 raw files with the package `xlrd`.

## Sources and Licenses

The workbooks are adapted from three public datasets. The processing above lists the changes.
Each workbook keeps the license of its source.

| Dataset | Source | License |
|---|---|---|
| D1 | Y. Zhang, Q. Tang, Y. Zhang, J. Wang, U. Stimming and A. A. Lee, "Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning", Zenodo, 2020, https://doi.org/10.5281/zenodo.3633835. Article: Nature Communications 11, 1706 (2020). | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| D2 | M. Rashid, M. Sheikh, J. Sansom, M. Faraji Niri and J. Marco, "DIB_Data", Mendeley Data, version 3, 2023, https://doi.org/10.17632/mn9fb7xdx6.3. Article: M. Rashid et al., Data in Brief 48, 109157 (2023). | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| D3 | H. Mustafa, C. Bourelly, M. Vitelli, F. Milano, M. Molinara and L. Ferrigno, "SoC Estimation on Li-ion Batteries: A New EIS-based Dataset for data-driven applications", Mendeley Data, version 2, 2024, https://doi.org/10.17632/cb887gkmxw.2. Article: Data in Brief 57, 110947 (2024). | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
