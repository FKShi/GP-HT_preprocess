# Double Cole-Cole Fitting and Bootstrap Statistics for Experimental Data

This folder contains `01_Experimental_Data_DoubleCole_Fitting_and_Bootstrap_Statistics.ipynb`, which fits experimental impedance spectra with the double Cole-Cole model and evaluates parameter stability using multi-start fitting, bootstrap resampling, profile likelihood, and paired statistics.

How to use:

1. The default example input is controlled by `DATASET_NAME`, which reads a cell, potato, or mouse example workbook from the sibling `6 Case Data` folder.
2. To reproduce the paired GP-versus-Exp comparison in the manuscript, set `INPUT_EXCEL` to the GP-HT summary workbook exported by Part 4. That workbook should contain `{raw/sparse/limited/noisy}_Freq_{Exp/GP}`, `Re`, and `Im` columns.
3. Check `OUTPUT_DIR` and change it to your preferred output directory if needed.
4. Adjust `N_RANDOM_STARTS`, `RUN_BOOTSTRAP`, `N_BOOTSTRAP`, and `RUN_PROFILE_LIKELIHOOD` according to the sample size and available computing resources.
5. Run the notebook from top to bottom. Outputs include fitted-parameter tables, bootstrap samples, profile likelihood results, paired Wilcoxon/FDR statistics, fitted-curve tables, and diagnostic figures.
