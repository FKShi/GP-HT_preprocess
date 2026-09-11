# GP-HT Processing of Experimental Data

This folder processes experimental impedance data with two notebooks: DRT-GPHT and DCT-GPHT. Example input data are stored in the sibling folder `6 Case Data`.

- `01_DRT_GPHT_experimental_data_processing.ipynb`: runs imInput GP-HT with the DRT kernel and exports reconstructed curves, credible intervals, and 4 x 3 comparison figures under different degraded input conditions.
- `02_DCT_GPHT_experimental_data_processing.ipynb`: runs imInput and reInput branches in the admittance domain with the DCT-form/unbounded kernel, then exports impedance-domain results and figures.
- `GP_hilbert.py`: DRT-kernel functions from the original GP-HT work by Ciucci et al.
- `GP_hilbert_unbounded.py`: DCT/unbounded-kernel functions based on the original GP-HT framework and the DCT formulation.

How to use:

1. Set `DATASET_NAME` in the parameter-configuration cell. Available values are `"cell"`, `"potato"`, and `"mouse"`.
2. The example input file is automatically selected from `../6 Case Data/case_data_cell.xlsx`, `case_data_potato.xlsx`, or `case_data_mouse.xlsx`.
3. Check `OUTPUT_ROOT` and change it to your preferred output directory if needed.
4. Adjust `NOISE_LEVELS`, `SPARSE_RATIOS`, `LIMITED_PERCENT_LIST`, and `LIMITED_RANGE_LIST` as needed. The default settings match the manuscript examples.
5. Run the notebook from top to bottom. Outputs include Excel data tables, optimization logs, a figure index, and PNG figures for each sample.
