#!/usr/bin/env python3

import nibabel as nib
import numpy as np


def simulate_volumes(real_vol_path, output_file):

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
