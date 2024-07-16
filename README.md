# FreeSurfer-FastSurfer_Comparison

This repository contains the scripts to perform the analysis and the comparison between brain segmentations. The segmentations are the ones obtained from different software packages (FreeSurfer and FastSurfer), starting from Magnetic Resonance Imaging (MRI) T1 scans. The aim is to provide an easy-to-use tool for comparing these segmentations. The comparison is performed on binary volumes extracted from the .mgz files. The results include common similarity metrics like Dice Similarity Coefficient, Jaccard Index, Volumetric Difference, and Hausdorff Distance, as well as visualizations of the differences slice by slice in each plane (axial, coronal and sagittal) and in a 3D volume. Bar plots and line plots are also generated to help users to easily identify the slices with the largest differences in the comparison.


## Installation and Dependencies

To clone the git repository, type the following commands in the terminal:
```
git clone https://github.com/Maddi5/FreeSurfer-FastSurfer_Comparison.git
cd FreeSurfer-FastSurfer_Comparison
```

### Dependencies

In order to run the code, the user needs to have the following packages installed:

* numpy
* nibabel
* pandas
* scipy
* scikit-image
* matplotlib
* seaborn
* medpy

These dependencies are automatically installed when running the main code (`segmentations_comparison.py`) through the call to the Setup_Script() function in the first lines.


## Usage
Place your .mgz files in the FreeSurfer-FastSurfer_Comparison directory. Then substitute 'FreeSurfer.mgz', 'FreeSurfer_auto.mgz' and 'FastSurfer.mgz' with the names of your files in the following part of `segmentations_comparison.py` script:

```
input_volumes= {
    'FreeSurfer': load_volume('FreeSurfer.mgz'),
    'FreeSurfer_auto': load_volume('FreeSurfer_auto.mgz'),
    'FastSurfer': load_volume('FastSurfer.mgz')
}

```
Then run `segmentations_comparison.py` to compute the metrics and save the results, and `plot_differences.py` to generate and save the visualizations. The results will be saved in the `Results` folder.


## Running tests

The `test_pytest.py` script contains the tests for all the functions. In order to run them, one has to install the pytest package, and then run the `pytest` command (being in the FreeSurfer-FastSurfer_Comparison directory):
```
pytest
```
The scripts were written in `Python 3.12.2` on `Windows 10`, and the functions were tested in the same environment.




## Repository Structure and Contents

The main directory includes:

- `segmentations_comparison.py`: Script to load segmentation volumes, extract the largest connected component, compute comparison metrics and difference matrix.
- `metrics.py`: Script containing functions to compute various metrics between two binary volumes.
- `plot_differences.py`: Script to generate visualizations (bar plots, line plots, 2D plots, 3D plots) of the differences between segmentation volumes.
- `setup.py`: script to check for and install the necessary packages for running all other scripts
- `test_pytest.py`: script containing all test functions


The [Simulated_Data](https://github.com/Maddi5/FreeSurfer-FastSurfer_Comparison/tree/main/Simulated_Data) folder includes the `simulate_volumes.py` script used to simulate the brain volume segmentations, starting from real MRI data. This was done to protect sensitive data while providing the user useful volumes to run the main code on. `Simulated_FreeSurfer.mgz`, `Simulated_FreeSurfer_auto.mgz`, `Simulated_FastSurfer.mgz` are example outputs from this code, ready to be used in the `segmentations_comparison.py` script.


The [Test_volumes](https://github.com/Maddi5/FreeSurfer-FastSurfer_Comparison/tree/main/Test_volumes) folder includes the `Create_test_volumes.py` script used to create test volumes for test functions in the `test_pytest.py` script. `empty_volume.mgz` and `test_input_volume.mgz` are outputs of this script, ready to be used in testing.


The [Results](https://github.com/Maddi5/FreeSurfer-FastSurfer_Comparison/tree/main/Results) folder contains examples for all the outputs from the scripts. This includes an Excel file with metrics results `results_metrics.xlsx`, difference matrices for each case (`difference_matrix_case0_vs_case1.npy`), bar plots of the total sum of differences in each view (`Total_sum_differences_view.png`), line plots with % difference (`% Different_pixel_view.png`), 2D difference matrices for each plane in each comparison (`Differences_case0_vs_case1_view.xlsx`), 3D difference matrix in each view (`Differences_matrix_3D_FreeSurfer_vs_FastSurfer_view.png`).
After running the `segmentations_comparison.py` and `plot_differences.py` scripts, the user will find all outputs saved in this folder.

The [Report](https://github.com/Maddi5/FreeSurfer-FastSurfer_Comparison/tree/main/Report) folder contains the report of a real-case analysis on FreeSurfer and FastSurfer segmentations from three patients, written for the "Pattern Recognition" exam at the University of Bologna.





## Scripts Overview

### metrics.py

This script contains functions to compute the following metrics between two binary volumes:

- **Jaccard Index**: Measures the overlap between two sets.
- **Volumetric Difference**: Measures the difference in volume between two sets.
- **Hausdorff Distance**: Measures the distance between points in two sets.


### segmentations_comparison.py

This script performs the following steps:

1. Load segmentation volumes from .mgz files
2. Extract the largest connected component.
3. Compute metrics (Dice Similarity Coefficient, Jaccard Index, Volumetric Difference, Hausdorff Distance), using the functions defined in `metrics.py` script.
4. Save the metrics into an Excel file.
5. Compute and save difference matrices.

Run the script:

```
bash
python segmentations_comparison.py

```

### plot_differences.py

This script generates visualizations of the differences between segmentation volumes using the difference matrices saved by `segmentations_comparison.py`:

- Bar plots of the total sum of differences for each slice in the three views.
- Line plots of the percentage of different pixels for each slice in each view.
- 2D plots of the difference matrices in each view in the slices selected by the user
- 3D plots of the difference matrices.

Run the script:

```
bash
python plot_differences.py
```
