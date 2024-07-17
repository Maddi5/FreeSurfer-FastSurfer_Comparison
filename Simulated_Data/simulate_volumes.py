#!/usr/bin/env python3

import nibabel as nib
import numpy as np


def simulate_volumes(real_vol_path, output_file):

    """
    Simulate new volume data by assigning random labels to the non-background regions of a real volume

    Parameters
    ----------
    real_vol_path : str
        Path to the input volume file in .mgz format
    output_file : str
        Path to save the simulated volume file in .mgz format

    Returns
    -------
    None
        The function saves the simulated volume as a new .mgz file

    Notes
    -----
    The function assigns random integer values between 1 and 10 to the non-background regions of the input volume.
    Background regions (where voxel value is 0) are not modified.

    """


    # Load the real volume
    real_vol = nib.load(real_vol_path).get_fdata()

    # Identify the non-background regions (where voxel is not 0)
    non_background = real_vol != 0

    # Assign random numbers between 1 and 10 to non-background regions, to simulate different labels
    real_vol[non_background] = np.random.randint(1, 10, size=np.count_nonzero(non_background))

    # Create new image
    simulated_image = nib.Nifti1Image(real_vol, nib.load(real_vol_path).affine)

    # Save new image
    nib.save(simulated_image, output_file)



#change the path
simulate_volumes('../mri/aseg.mgz', 'Simulated_FreeSurfer.mgz')
simulate_volumes('../mri/aseg.auto.mgz', 'Simulated_FreeSurfer_auto.mgz')
simulate_volumes('../mri/aseg.mgz', 'Simulated_FastSurfer.mgz')
