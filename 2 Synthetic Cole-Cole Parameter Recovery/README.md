# Cole-Cole Parameter Recovery for Synthetic Spectra

This folder reads processed synthetic spectra and refits parametric models to evaluate parameter-recovery errors. Shared functions are stored in `synthetic_recovery_v3_common.py`. The six notebooks cover two model types and three processing methods:

- `01_DoubleCole_GPHT_parameter_recovery.ipynb`: recovers model parameters from the GP-HT-processed double Cole-Cole spectra.
- `02_DoubleCole_BHT_parameter_recovery.ipynb`: recovers model parameters from the BHT-processed double Cole-Cole spectra.
- `03_DoubleCole_LinKK_parameter_recovery.ipynb`: recovers model parameters from the Lin-KK-processed double Cole-Cole spectra.
- `04_RctZARC_GPHT_parameter_recovery.ipynb`: recovers model parameters from the GP-HT-processed Rct+ZARC spectra.
- `05_RctZARC_BHT_parameter_recovery.ipynb`: recovers model parameters from the BHT-processed Rct+ZARC spectra.
- `06_RctZARC_LinKK_parameter_recovery.ipynb`: recovers model parameters from the Lin-KK-processed Rct+ZARC spectra.

How to use:

1. Check the input workbook path `PRIMARY_WORKBOOK`. If it does not exist, the code will try to read `FALLBACK_WORKBOOK` from the current notebook folder.
2. Replace the input path according to your data source, then run the notebook from top to bottom.
3. The output directory is controlled by `OUTPUT_DIR`. Results include parameter-fit tables, parameter APE, curve errors, full-frequency-grid errors, and summary figures.
4. The shared module provides model formulas, error metrics, and plotting functions for all notebooks. Keep the expected column names consistent when editing the workflow.
