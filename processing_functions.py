"""

Script with the functions to select the slices for one specific view and to process them

Author: Maddalena Cavallo
Date: July 2024

"""

import numpy as np


def slice_selection(difference_matrix, plane, i):
     
    """
    This function selects a 2D slice from the 3D difference matrix based on the specified plane and index.

    Parameters
    ----------
    difference_matrix : numpy.ndarray
        A 3D numpy array representing the difference matrix between two segmentations
    plane : str
        The plane along which to move to "cut" the brain and extract the slice. Should be one of 'Axial', 'Coronal', or 'Sagittal'
    i : int
        The index of the slice along the specified plane.

    Returns
    -------
    slice : numpy.ndarray
        A 2D numpy array representing the selected slice from the difference matrix provided as input
    """
	
    #select in which direction to move to "cut" the brain
    if plane == 'Axial':
        slice = difference_matrix[i, :, :]
    elif plane == 'Coronal':
        slice = difference_matrix[:, i, :]
    elif plane == 'Sagittal':
        slice = difference_matrix[:, :, i]

    return slice



def slice_process(slice):

    """
    This function processes a 2D slice from the 3D difference matrix, computing the total number of different pixels, the percentage and the total sum of the difference matrix in that slice

    Parameters
    ----------
    slice : numpy.ndarray
        A 2D numpy array representing a slice from the difference matrix.

    Returns
    -------
    slice_num_diff : int
        The total number of different pixels in the slice
    perc_diff : float
        The percentage of different pixels in the slice.
    slice_sum : int
        The total sum of the difference matrix values in the slice
    """

    # Sum of the difference matrix in that slice
    slice_sum = np.sum(slice)

    # Apply the absolute value to compute the total number of different pixels in that slice
    slice_num_diff = np.sum(np.abs(slice))

    # Compute the percentage of different pixels in that slice
    slice_total_pixels = slice.shape[0] * slice.shape[1]  # take always the same axis because it's symmetric in 3D
    perc_diff = (slice_num_diff / slice_total_pixels) * 100

    return (slice_num_diff, perc_diff, slice_sum)