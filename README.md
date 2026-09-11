# GP-HT_preprocess

Companion code and data for the manuscript:

**"Enhanced Impedimetric Analysis under Constrained Measurement Conditions through Gaussian Process Hilbert Transform"**
*Electrochimica Acta* (revised manuscript, Manuscript No. EA26-5054R)

This repository contains the complete analysis code, the synthetic-spectrum generators, and representative experimental data used in the study. The workflow systematically evaluates the Gaussian Process Hilbert Transform (GP-HT) framework of Ciucci and co-workers as a preprocessing step for electrochemical impedance spectroscopy (EIS) data acquired under noisy, sparse, and bandwidth-limited conditions. Evaluation is performed at two levels — curve reconstruction and downstream Cole–Cole parameter estimation — on synthetic spectra with known ground truth and on experimental spectra (cell monolayers, potato tissue, in vivo mouse muscle, and a lithium-ion battery).

## Repository structure

| Folder | Contents |
|---|---|
| `1 Synthetic Spectra Generation and GP-HT Processing` | Generators for the double Cole–Cole, Rct+ZARC, and 2ZARC+Warburg synthetic spectra; dual-branch (imInput/reInput) GP-HT processing under raw, noisy, sparse, and limited input conditions; shared DRT-kernel module `GP_hilbert.py`. |
| `2 Synthetic Cole-Cole Parameter Recovery` | Parameter refitting of GP-HT-, BHT-, and Lin-KK-processed synthetic spectra (six notebooks) with curve- and parameter-recovery error metrics; shared module `synthetic_recovery_v3_common.py`. |
| `3 non-KK Synthetic Spectra Method Comparison` | Synthetic spectra with a controlled low-frequency non-KK drift; GP-HT vs. BHT vs. Lin-KK reconstruction comparison and summary notebook; shared module `nonkk_common.py`. |
| `4 Experimental Data GP-HT Processing` | GP-HT processing of the experimental spectra with the DRT kernel (`01_...`) and with the DCT/unbounded kernel (`02_...`, the DCT-kernel sensitivity analysis reported in Supplementary Fig. S7); kernel modules `GP_hilbert.py` and `GP_hilbert_unbounded.py`. |
| `5 Experimental Data Double Cole-Cole Fitting` | Double Cole–Cole fitting of experimental and GP-HT-reconstructed spectra with multi-start optimization, bootstrap parameter-uncertainty estimation, profile likelihood, and paired Wilcoxon/BH-FDR statistics. |
| `6 Case Data` | Representative raw experimental spectra: 15 cell-monolayer spectra (`case_data_cell.xlsx`), 3 potato-tissue spectra (`case_data_potato.xlsx`), and 3 in vivo mouse-muscle spectra (`case_data_mouse.xlsx`). Each sheet holds one independent spectrum with columns `raw_Freq_Exp`, `raw_Re_Exp`, `raw_Im_Exp`. |

Each folder contains its own `README.md` with detailed run instructions and parameter descriptions. The repository root additionally provides `requirements.txt` (environment specification) and `download_battery_data.py` (fetches the public battery spectrum, see below).

## Reproduction workflow

Run the notebooks in the following order (see the folder-level READMEs for configurable parameters):

1. **Folder 1** — generate the synthetic spectra and produce the dual-branch GP-HT reconstructions under the raw, noisy, sparse, and limited input conditions.
2. **Folder 2** — refit the processed synthetic spectra to quantify parameter-recovery errors (GP-HT vs. BHT vs. Lin-KK).
3. **Folder 3** — generate the non-KK synthetic datasets (notebooks `01`–`04` in order) and summarize the three-method comparison (`05`).
4. **Folder 4** — process the experimental spectra with the DRT kernel; optionally run the DCT-kernel sensitivity analysis.
5. **Folder 5** — fit the double Cole–Cole model to the experimental and reconstructed spectra and compute bootstrap and paired statistics. For the paired GP-versus-Exp comparison reported in the manuscript, point `INPUT_EXCEL` to the summary workbook exported by Folder 4.

All notebooks write their outputs (Excel tables and PNG figures) to folder-local output directories controlled by `OUTPUT_DIR`/`OUTPUT_ROOT`.

## Environment

- Python ≥ 3.9 with JupyterLab.
- Install the required packages with:

```bash
pip install -r requirements.txt
jupyter lab
```

## Data availability

- **Cell monolayers / potato tissue / mouse muscle:** the workbooks in `6 Case Data` contain representative raw spectra (the majority of the cell-monolayer spectra, and three independent raw spectra each for potato tissue and in vivo mouse muscle). Running the notebooks in Folders 4 and 5 on these inputs reproduces the corresponding pipeline outputs shown in the manuscript.
- **Lithium-ion battery:** the battery spectrum analyzed in the manuscript is taken from the public dataset of Mustafa et al., *Data in Brief* 57, 110947 (2024), dataset DOI: [10.17632/cb887gkmxw](https://data.mendeley.com/datasets/cb887gkmxw) (Mendeley Data), article DOI: [10.1016/j.dib.2024.110947](https://doi.org/10.1016/j.dib.2024.110947). The exact file used is `Hk_IFR14500_SoC_5_04-07-2023_05-13.csv` (abbreviated in the manuscript as `Hk_IFR14500_SoC_5_04-07-2023_05`; LFP cell, 5% state of charge, 0.01–1000 Hz, 28 frequency points, four-terminal measurement). Run `python download_battery_data.py` from the repository root to fetch this file into `6 Case Data`; the script verifies the SHA-256 checksum published by Mendeley Data.
- The remaining tissue spectra are part of a pulsed-field-ablation assessment database subject to collaboration agreements with industrial partners and are available from the corresponding authors upon reasonable request and with permission of the partner institutions.

## Attribution and citations

If you use this code or data, please cite the manuscript above. The GP-HT kernel and posterior-prediction functions in `GP_hilbert.py` are derived from the companion code of the original GP-HT work:

- F. Ciucci, "The Gaussian Process Hilbert Transform (GP-HT): Testing the Consistency of Electrochemical Impedance Spectroscopy Data," *Journal of The Electrochemical Society* 167, 126503 (2020). DOI: [10.1149/1945-7111/aba937](https://doi.org/10.1149/1945-7111/aba937); original code: [github.com/ciuccislab/GP-HT](https://github.com/ciuccislab/GP-HT).

The DCT-kernel implementation in `GP_hilbert_unbounded.py` builds on the distribution-of-capacitive-times formulation:

- B. Py, A. Maradesa, and F. Ciucci, "From theory to practice: Unlocking the distribution of capacitive times in electrochemical impedance spectroscopy," *Electrochimica Acta* 479, 143741 (2024). DOI: [10.1016/j.electacta.2023.143741](https://doi.org/10.1016/j.electacta.2023.143741).

The battery spectrum is from:

- H. Mustafa, C. Bourelly, M. Vitelli, F. Milano, M. Molinara, and L. Ferrigno, "SoC estimation on Li-ion batteries: A new EIS-based dataset for data-driven applications," *Data in Brief* 57, 110947 (2024). DOI: [10.1016/j.dib.2024.110947](https://doi.org/10.1016/j.dib.2024.110947).

## Contact

- Fukun Shi, Ph.D. (corresponding author): fukunshi@sibet.ac.cn
- Jie Zhuang, Ph.D. (corresponding author): jzhuang@sibet.ac.cn

Suzhou Institute of Biomedical Engineering and Technology, Chinese Academy of Sciences
