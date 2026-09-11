# Comparison of Different Methods on non-KK Synthetic Spectra

This folder constructs synthetic impedance spectra with a low-frequency non-KK drift and compares three processing methods: GP-HT, BHT, and Lin-KK. Shared functions are stored in `nonkk_common.py`.

- `01_nonKK_synthetic_spectra_generation.ipynb`: generates non-KK datasets for the double Cole-Cole and Rct+ZARC models and saves raw/noisy/sparse/limited inputs.
- `02_nonKK_GPHT_reconstruction.ipynb`: applies GP-HT-like reconstruction to non-KK input spectra.
- `03_nonKK_BHT_reconstruction.ipynb`: applies BHT-like reconstruction to non-KK input spectra.
- `04_nonKK_LinKK_reconstruction.ipynb`: applies Lin-KK-like reconstruction to non-KK input spectra.
- `05_nonKK_results_summary.ipynb`: summarizes curve errors, parameter errors, and comparison figures for the three methods.

How to use:

1. Start JupyterLab from this folder if possible, and run notebooks `01` to `05` in order.
2. Outputs are saved by default to `nonkk_method_comparison_outputs` under the current folder.
3. To change noise, sparsity, or truncation settings, edit the condition parameters in `nonkk_common.py` or in the corresponding notebook.
4. The three methods use the same input data and the same error metrics, which allows direct comparison of their behavior under non-KK perturbations.
