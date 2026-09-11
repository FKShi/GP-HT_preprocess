# Synthetic Spectra Generation and Dual-Branch GP-HT Processing

This folder generates three controllable synthetic impedance spectra and processes them with the dual-branch GP-HT workflow under raw, noisy, sparse, and limited input conditions.

- `01_DoubleCole_GPHT_dual_branch_processing.ipynb`: generates a double Cole-Cole synthetic spectrum and compares the imInput and reInput GP-HT reconstruction branches.
- `02_RctZARC_GPHT_dual_branch_processing.ipynb`: generates an Rct+ZARC synthetic spectrum for GP-HT reconstruction tests on an electrochemical impedance model.
- `03_2ZARC_Warburg_GPHT_dual_branch_processing.ipynb`: generates a 2ZARC+Warburg synthetic spectrum for GP-HT reconstruction tests on spectra with a diffusion tail.
- `GP_hilbert.py`: DRT-kernel GP-HT and posterior-prediction functions. The theory and code are based on the original GP-HT work by Ciucci et al. (J. Electrochem. Soc. 2020, DOI: 10.1149/1945-7111/aba9c0).

How to use:

1. Open the corresponding notebook and check whether `OUTPUT_DIR` is the desired output location.
2. Modify `NOISE_LEVELS`, `SPARSE_RATIOS`, `LIMITED_RANGE_LIST`, and `LIMITED_PERCENT_LIST` as needed.
3. To adjust GP-HT smoothing and noise assumptions, edit the `SIGMA_DRT_*`, `SIGMA_SB_*`, `ELL_*`, `SIGMA_N_*`, and `TAU_MAX_*` entries in `GPHT_CONFIG`.
4. Run the notebook from top to bottom. The outputs usually include synthetic spectra, degraded input spectra, GP-HT predictions, error metrics, and figures.
