"""

Script with the functions to select the slices for one specific view and to process them

Author: Maddalena Cavallo
Date: July 2024

"""

import numpy as np


def slice_selection(difference_matrix, plane, i):
	
    #select in which direction to move to "cut" the brain
    if plane == 'Axial':
        slice = difference_matrix[i, :, :]
    elif plane == 'Coronal':
        slice = difference_matrix[:, i, :]
    elif plane == 'Sagittal':
        slice = difference_matrix[:, :, i]

    return slice



def slice_process(slice):

    # Sum of the difference matrix in that slice
    slice_sum = np.sum(slice)

    # Apply the absolute value to compute the total number of different pixels in that slice
    slice_num_diff = np.sum(np.abs(slice))

    # Compute the percentage of different pixels in that slice
    slice_total_pixels = slice.shape[0] * slice.shape[1]  # take always the same axis because it's symmetric in 3D
    perc_diff = (slice_num_diff / slice_total_pixels) * 100

    return (slice_num_diff, perc_diff, slice_sum)